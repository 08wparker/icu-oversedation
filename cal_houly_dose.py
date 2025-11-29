# Init

import duckdb
import pandas as pd
from datetime import datetime
from clifpy.utils.unit_converter import convert_dose_units_by_med_category
from clifpy import MedicationAdminContinuous, Vitals
from clifpy.utils import apply_outlier_handling


# Utils

def remove_meds_duplicates(meds_df: pd.DataFrame) -> pd.DataFrame:
    if 'mar_action_category' not in meds_df.columns: 
        print('mar_action_category not available, deduping by mar_action_name instead')
        q = f"""
        SELECT *
        FROM meds_df
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY hospitalization_id, admin_dttm, med_category
            ORDER BY 
                -- apply mar action dedup logic
                CASE WHEN mar_action_name IS NULL THEN 10
                    WHEN regexp_matches(mar_action_name, 'verify', 'i') THEN 9
                    WHEN regexp_matches(mar_action_name, '(stopped)|(held)|(paused)|(completed)', 'i') THEN 8
                    ELSE 1 END,
                -- if tied at the same mar action, deprioritize zero or null doses
                CASE WHEN med_dose > 0 THEN 1
                    ELSE 2 END,
                -- prioritize larger doses
                med_dose DESC 
        ) = 1
        ORDER BY hospitalization_id, med_category, admin_dttm;
        """
    else:
        q = f"""
        SELECT *
        FROM meds_df
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY hospitalization_id, admin_dttm, med_category
            ORDER BY 
                -- apply mar action dedup logic
                CASE WHEN mar_action_category IS NULL THEN 10
                    WHEN mar_action_category in ('verify', 'not_given') THEN 9
                    WHEN mar_action_category = 'stop' THEN 8
                    WHEN mar_action_category = 'going' THEN 7
                    ELSE 1 END,
                -- if tied at the same mar action, deprioritize zero or null doses
                CASE WHEN med_dose > 0 THEN 1
                    ELSE 2 END,
                -- prioritize larger doses
                med_dose DESC 
        ) = 1
        ORDER BY hospitalization_id, med_category, admin_dttm;
        """
    return duckdb.sql(q).to_df()


# Fetch all MAR of continuous sedative admin

# placeholder example
cohort_hosp_ids = ['H00001', 'H00002', 'H00003']


# get a df with just patient's weights from the vitals table for calculating weight-based med dosing
vitals = Vitals.from_file(
    config_path = 'config/config.json',
    columns = ['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
    filters = {
        'vital_category': ['weight_kg'],
        'hospitalization_id': cohort_hosp_ids
    }
    )
apply_outlier_handling(vitals, outlier_config_path = 'config/outlier_config.yaml')
pt_weights_df = vitals.df


# load the continuous sedation data
cont_sed = MedicationAdminContinuous.from_file(
    config_path = 'config/config.json',
    columns = [
        'hospitalization_id', 'admin_dttm', 'med_name', 'med_category', 'med_dose', 'med_dose_unit',
        'mar_action_name', 'mar_action_category'
        ],
    filters = {
        'med_category': ['hydromorphone', 'fentanyl', 'lorazepam', 'midazolam', 'propofol'], 
        'hospitalization_id': cohort_hosp_ids
    }
    )

cont_sed_deduped = remove_meds_duplicates(cont_sed.df)
n_removed = len(cont_sed.df) - len(cont_sed_deduped)
print(f"Removed {n_removed} ({n_removed / len(cont_sed.df):.2%}) duplicates by MAR action")

cont_sed_preferred_units = {
    'propofol': 'mg/min',
    'midazolam': 'mg/min',
    'fentanyl': 'mcg/min',
    'hydromorphone': 'mg/min',
    'lorazepam': 'mg/min'
    }
cont_sed_converted, cont_sed_convert_summary = convert_dose_units_by_med_category(
    cont_sed_deduped,
    vitals_df = pt_weights_df,
    preferred_units = cont_sed_preferred_units,
    override = True
)

cont_sed_converted.rename(columns={
    'med_dose': 'med_dose_original', 
    'med_dose_unit': 'med_dose_unit_original', 
    'med_dose_converted': 'med_dose', 
    'med_dose_unit_converted': 'med_dose_unit'
    }, inplace=True)

cont_sed.df = cont_sed_converted

# apply outlier handling
apply_outlier_handling(cont_sed, outlier_config_path = 'config/outlier_config.yaml')
cont_sed_converted = cont_sed.df


# converting to wide format with unit-aware column names
q = """
WITH t1 AS (
    SELECT hospitalization_id
        , admin_dttm as event_dttm
        , med_category_unit: med_category || '_' || REPLACE(med_dose_unit, '/', '_') || '_cont'
        , med_dose
    FROM cont_sed_converted
)
, t2 AS (
    PIVOT_WIDER t1
    ON med_category_unit
    USING FIRST(med_dose)
)
SELECT *
FROM t2
ORDER BY hospitalization_id, event_dttm
"""
cont_sed_w = duckdb.sql(q).df()


# Build hourly grid of timestamps

# cohort_start_end_dttms is a dataframe with 3 columns: the id column (here hospitalization_id), _start_dttm, _end_dttm
# where _start_dttm and _end_dttm are the first and last timestamps charted for that hospitalization_id -- used to define the observation window to be broken down into hourly grids
# the following is just a placeholder example

cohort_start_end_dttms = pd.DataFrame({
    'hospitalization_id': [1, 2, 3],
    '_start_dttm': [datetime(2021, 1, 1, 0, 0, 0), datetime(2021, 1, 1, 0, 0, 0), datetime(2021, 1, 1, 0, 0, 0)],
    '_end_dttm': [datetime(2021, 1, 1, 23, 59, 59), datetime(2021, 1, 1, 23, 59, 59), datetime(2021, 1, 1, 23, 59, 59)]
})


cohort_start_end_dttms['_start_hr'] = cohort_start_end_dttms['_start_dttm'].dt.floor('h', ambiguous='NaT')
cohort_start_end_dttms['_end_hr'] = cohort_start_end_dttms['_end_dttm'].dt.ceil('h', ambiguous='NaT')

q = """
SELECT 
    hospitalization_id,
    unnest(generate_series(_start_hr, _end_hr, INTERVAL '1 hour')) AS event_dttm
FROM cohort_start_end_dttms
ORDER BY hospitalization_id, event_dttm
"""
cohort_hrly_grids = duckdb.sql(q).df()


# Merge and calculate hourly cumulative dose

q = """
-- create the hourly grid for the wide sedation table
FROM cohort_hrly_grids g
FULL JOIN cont_sed_w m USING (hospitalization_id, event_dttm)
ORDER BY hospitalization_id, event_dttm
"""
# wide table with hourly grids inserted
cont_sed_wg = duckdb.sql(q).df()
cont_sed_wg['_dh'] = cont_sed_wg['event_dttm'].dt.floor('h', ambiguous='NaT')
cont_sed_wg['_hr'] = cont_sed_wg['event_dttm'].dt.hour
print(len(cont_sed_wg))


q = """
WITH t1 AS (
    FROM cont_sed_wg g
    SELECT hospitalization_id, event_dttm, _dh, _hr
        , LAST_VALUE(COLUMNS('_cont') IGNORE NULLS) OVER (
            PARTITION BY hospitalization_id ORDER BY event_dttm
        )
        , _duration: EXTRACT(EPOCH FROM (LEAD(event_dttm, 1, event_dttm) OVER w - event_dttm)) / 60.0
    WINDOW w AS (PARTITION BY hospitalization_id ORDER BY event_dttm)
), t2 AS (
    FROM t1
    SELECT hospitalization_id, event_dttm, _dh, _hr, _duration
        --, COALESCE(_duration_mins, 0)
        , COALESCE(COLUMNS('_cont'), 0) 
), t3 AS (
    FROM t2
    SELECT hospitalization_id, event_dttm, _dh, _hr, _duration
        , COLUMNS('_cont') * _duration
), t4 AS (
    FROM t3
    SELECT hospitalization_id, _dh, _hr
        , SUM(COLUMNS('_cont'))
    GROUP BY hospitalization_id, _dh, _hr
)
SELECT *
FROM t4
-- ORDER BY hospitalization_id, event_dttm
ORDER BY hospitalization_id, _dh
"""
cont_sed_dose_by_hr = duckdb.sql(q).df()


