# Load required libraries
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import pyarrow.parquet as pq
import time

# Add the parent directory of the current script to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import config after adding parent to path
from utils.config import config

# Access configuration parameters
site_name = config['site_name']
tables_path = config['tables_path']
file_type = config['file_type']


# Print the configuration parameters
print(f"Site Name: {site_name}")
print(f"Tables Path: {tables_path}")
print(f"File Type: {file_type}")

# Cohort identification script for ICU oversedation study
# Objective: Identify adult patients who received invasive mechanical ventilation (IMV) for >= 24 hours
#
# Inclusion criteria:
# 1. Adults (>= 18 years old)
# 2. Invasive mechanical ventilation for >= 24 hours
#
# Output:
# - cohort_ids: List of hospitalization_id meeting criteria
# - cohort_data: Filtered CLIF tables for the cohort
# - cohort_summary: Summary statistics describing the cohort

# Specify cohort parameters

## Date range (optional - can be None to include all dates)
start_date = None  # "2020-01-01" or None for no date filter
end_date = None    # "2021-12-31" or None for no date filter

## Minimum IMV duration in hours
min_imv_hours = 4

## Confirm that these are the correct paths
adt_filepath = f"{tables_path}/clif_adt.{file_type}"
hospitalization_filepath = f"{tables_path}/clif_hospitalization.{file_type}"
vitals_filepath = f"{tables_path}/clif_vitals.{file_type}"
labs_filepath = f"{tables_path}/clif_labs.{file_type}"
meds_filepath = f"{tables_path}/clif_medication_admin_continuous.{file_type}"
resp_support_filepath = f"{tables_path}/clif_respiratory_support.{file_type}"


def read_data(filepath, filetype):
    """
    Read data from file based on file type.
    Parameters:
        filepath (str): Path to the file.
        filetype (str): Type of the file ('csv' or 'parquet').
    Returns:
        DataFrame: DataFrame containing the data.
    """
    start_time = time.time()  # Record the start time
    file_name = os.path.basename(filepath)
    if filetype == 'csv':
        df = pd.read_csv(filepath)
    elif filetype == 'parquet':
        table = pq.read_table(filepath)
        df = table.to_pandas()
    else:
        raise ValueError("Unsupported file type. Please provide either 'csv' or 'parquet'.")

    end_time = time.time()  # Record the end time
    load_time = end_time - start_time  # Calculate the loading time

    # Calculate the size of the loaded dataset in MB
    dataset_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"File name: {file_name}")
    print(f"Time taken to load the dataset: {load_time:.2f} seconds")
    print(f"Size of the loaded dataset: {dataset_size_mb:.2f} MB\n")

    return df


clif_adt = read_data(adt_filepath, file_type)
clif_hospitalization = read_data(hospitalization_filepath, file_type)
clif_vitals = read_data(vitals_filepath, file_type)
clif_labs = read_data(labs_filepath, file_type)
clif_medication_admin_continuous = read_data(meds_filepath, file_type)
clif_respiratory_support = read_data(resp_support_filepath, file_type)


print("="*80)
print("COHORT IDENTIFICATION")
print("="*80)

# Step 1: Identify adults (>= 18 years old)
print("\nStep 1: Filtering for adults (age >= 18)")
clif_hospitalization['admission_dttm'] = pd.to_datetime(clif_hospitalization['admission_dttm'])

adults = clif_hospitalization[clif_hospitalization['age_at_admission'] >= 18].copy()

# Optional: Apply date range filter
if start_date is not None and end_date is not None:
    print(f"Applying date filter: {start_date} to {end_date}")
    adults = adults[
        (adults['admission_dttm'] >= start_date) &
        (adults['admission_dttm'] <= end_date)
    ]

print(f"Total adult hospitalizations: {len(adults)}")

# Step 2: Identify IMV episodes and calculate duration
print(f"\nStep 2: Identifying invasive mechanical ventilation (IMV) episodes")
print(f"Minimum IMV duration: {min_imv_hours} hours")

# Convert datetime
clif_respiratory_support['recorded_dttm'] = pd.to_datetime(clif_respiratory_support['recorded_dttm'])

# Identify IMV records
# IMV is typically indicated by device_category containing terms like:
# 'invasive', 'invasive ventilator', 'mechanical ventilator', 'endotracheal', etc.
# You may need to adjust this based on your data's specific values
imv_records = clif_respiratory_support[
    clif_respiratory_support['device_category'].str.contains(
        'IMV',
        case=False,
        na=False
    )
].copy()

print(f"Total IMV records: {len(imv_records)}")

# Calculate IMV duration for each hospitalization
imv_duration = imv_records.groupby('hospitalization_id').agg(
    first_imv=('recorded_dttm', 'min'),
    last_imv=('recorded_dttm', 'max'),
    num_imv_records=('recorded_dttm', 'count')
).reset_index()

# Calculate duration in hours
imv_duration['imv_duration_hours'] = (
    imv_duration['last_imv'] - imv_duration['first_imv']
).dt.total_seconds() / 3600

print(f"Hospitalizations with any IMV: {len(imv_duration)}")

# Step 3: Filter for IMV >= 24 hours
imv_24h = imv_duration[imv_duration['imv_duration_hours'] >= min_imv_hours].copy()
print(f"Hospitalizations with IMV >= {min_imv_hours} hours: {len(imv_24h)}")

# Step 4: Combine criteria - adults AND IMV >= 24 hours
cohort_ids = adults[
    adults['hospitalization_id'].isin(imv_24h['hospitalization_id'])
]['hospitalization_id'].unique()

print(f"\nFinal cohort size: {len(cohort_ids)} hospitalizations")

# Create cohort summary
cohort_summary = adults[adults['hospitalization_id'].isin(cohort_ids)].merge(
    imv_24h[['hospitalization_id', 'first_imv', 'last_imv', 'imv_duration_hours', 'num_imv_records']],
    on='hospitalization_id',
    how='left'
)

print("\nCohort Summary Statistics:")
print(f"Age - Mean: {cohort_summary['age_at_admission'].mean():.1f}, "
      f"Median: {cohort_summary['age_at_admission'].median():.1f}, "
      f"Range: [{cohort_summary['age_at_admission'].min():.0f}, {cohort_summary['age_at_admission'].max():.0f}]")
print(f"IMV Duration (hours) - Mean: {cohort_summary['imv_duration_hours'].mean():.1f}, "
      f"Median: {cohort_summary['imv_duration_hours'].median():.1f}, "
      f"Range: [{cohort_summary['imv_duration_hours'].min():.1f}, {cohort_summary['imv_duration_hours'].max():.1f}]")

# Step 5: Filter CLIF tables to cohort
print("\nStep 5: Filtering CLIF tables to cohort hospitalizations")

cohort_hospitalization = clif_hospitalization[
    clif_hospitalization['hospitalization_id'].isin(cohort_ids)
]
cohort_respiratory_support = clif_respiratory_support[
    clif_respiratory_support['hospitalization_id'].isin(cohort_ids)
]
cohort_vitals = clif_vitals[
    clif_vitals['hospitalization_id'].isin(cohort_ids)
]
cohort_labs = clif_labs[
    clif_labs['hospitalization_id'].isin(cohort_ids)
]
cohort_medication_admin_continuous = clif_medication_admin_continuous[
    clif_medication_admin_continuous['hospitalization_id'].isin(cohort_ids)
]

print(f"Cohort hospitalization records: {len(cohort_hospitalization)}")
print(f"Cohort respiratory support records: {len(cohort_respiratory_support)}")
print(f"Cohort vitals records: {len(cohort_vitals)}")
print(f"Cohort labs records: {len(cohort_labs)}")
print(f"Cohort medication records: {len(cohort_medication_admin_continuous)}")

# Step 6: Export cohort data
print("\nStep 6: Exporting cohort data")
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Export cohort IDs
cohort_ids_df = pd.DataFrame({'hospitalization_id': cohort_ids})
cohort_ids_df.to_csv(f"{output_dir}/cohort_ids_{site_name}.csv", index=False)
print(f"Saved: {output_dir}/cohort_ids_{site_name}.csv")

# Export cohort summary
cohort_summary.to_csv(f"{output_dir}/cohort_summary_{site_name}.csv", index=False)
print(f"Saved: {output_dir}/cohort_summary_{site_name}.csv")


# Export hospitalization IDs and first ventilation time to a separate file
cohort_imv_times = imv_24h[imv_24h['hospitalization_id'].isin(cohort_ids)][
    ['hospitalization_id', 'first_imv']
]

cohort_imv_times.to_csv(f"{output_dir}/cohort_imv_times_{site_name}.csv", index=False)
print(f"Saved: {output_dir}/cohort_imv_times_{site_name}.csv")



# Export filtered CLIF tables
cohort_hospitalization.to_csv(f"{output_dir}/cohort_hospitalization_{site_name}.csv", index=False)
cohort_respiratory_support.to_csv(f"{output_dir}/cohort_respiratory_support_{site_name}.csv", index=False)
cohort_vitals.to_csv(f"{output_dir}/cohort_vitals_{site_name}.csv", index=False)
cohort_labs.to_csv(f"{output_dir}/cohort_labs_{site_name}.csv", index=False)
cohort_medication_admin_continuous.to_csv(f"{output_dir}/cohort_medication_admin_continuous_{site_name}.csv", index=False)

print(f"\nAll cohort data exported to {output_dir}/")
print("="*80)
