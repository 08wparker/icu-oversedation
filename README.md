# Characterizing phenotypes of overuse of sedation and analgesia in mechanically ventilated patients

## Objective

This project aims to characterize the incidence and risk factors for overuse of sedation and analgesia in mechanically ventilated adult patients. 

## Methods

Observational retrospective cohort study of critically ill adults using real-world electronic health record (EHR) data standardized into the Common Longitudinal ICU data Format (CLIF).

#### Cohort identification

Adults (>= 18 years old) treated with invasive mechanical ventilation for > 24 hours

### Primary outcome: deep sedation
* Continuous infusion of a sedative or analgesic plus a RASS < -3


#### Richmond Agitation–Sedation Scale (RASS): clinical measure of depth of sedation. 

| RASS Score | Descriptor                       | Clinical Description                                                |
|------------|----------------------------------|---------------------------------------------------------------------|
| +4         | Combative                        | Overtly combative; violent; immediate danger to staff               |
| +3         | Very agitated                    | Pulls or removes tubes/catheters; aggressive                        |
| +2         | Agitated                         | Frequent non-purposeful movement; fights ventilator                 |
| +1         | Restless                         | Anxious but movements not aggressive or vigorous                    |
| 0          | Alert and calm                   | Normal level of consciousness                                       |
| –1         | Drowsy                           | Not fully alert; awakens to voice (eye opening & contact >10 sec)   |
| –2         | Light sedation                   | Briefly awakens to voice (eye opening & contact <10 sec)            |
| –3         | Moderate sedation                | Movement or eye opening to voice but no eye contact                 |
| –4         | Deep sedation                    | No response to voice; movement or eye opening to physical stimulus  |
| –5         | Unarousable                      | No response to voice or physical stimulation                        |


#### Analgesic and sedative medication list

* Both continuous infusions from `med_admin_continuous` and bolus doses from `med_admin_intermittent` in `med_group` == "sedation"



#### Indications for deep sedation

Easy to identify:
* Neuromuscular blockade infusion (infusion of cisatricurium, rocuronium, atracurium, vecuronium, succinylcholine)

More challenging:
* Uncontrolled status epilepticus (data from most recent EEG)
* Uncontrolled elevated intracranial pressure (>20cm H2O)
* Therapeutic hypothermia
* Organ donor


### Longitudinal sedation phenotypes

Unsupervised/self-supervised ML phenotyping of longitudinal sedation patterns


## CLIF VERSION 2.1 required tables and fields

Please refer to the online [CLIF data dictionary](https://clif-consortium.github.io/website/data-dictionary.html), [ETL tools](https://github.com/clif-consortium/CLIF/tree/main/etl-to-clif-resources), and [specific table contacts](https://github.com/clif-consortium/CLIF?tab=readme-ov-file#relational-clif) for more information on constructing the required tables and fields. 

*List all required tables for the project here, and provide a brief rationale for why they are required.*

Example:
The following tables are required:
1. **patient**: `patient_id`, `race_category`, `ethnicity_category`, `sex_category`
2. **hospitalization**: `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`, `age_at_admission`
3. **vitals**: `hospitalization_id`, `recorded_dttm`, `vital_category`, `vital_value`
   - `vital_category` = 'heart_rate', 'resp_rate', 'sbp', 'dbp', 'map', 'resp_rate', 'spo2'
4. **labs**: `hospitalization_id`, `lab_result_dttm`, `lab_category`, `lab_value`
   - `lab_category` = 'lactate'
5. **medication_admin_continuous**: `hospitalization_id`, `admin_dttm`, `med_name`, `med_category`, `med_dose`, `med_dose_unit`
   - `med_category` = "norepinephrine", "epinephrine", "phenylephrine", "vasopressin", "dopamine", "angiotensin", "nicardipine", "nitroprusside", "clevidipine", "cisatracurium"
6. **respiratory_support**: `hospitalization_id`, `recorded_dttm`, `device_category`, `mode_category`, `tracheostomy`, `fio2_set`, `lpm_set`, `resp_rate_set`, `peep_set`, `resp_rate_obs`


## Expected Results


*Describe the output of the analysis. The final project results should be saved in the [`output/final`](output/README.md) directory.*

## Detailed Instructions for running the project

## 1. Update `config/config.json`
Follow instructions in the [config/README.md](config/README.md) file for detailed configuration steps.

## 2. Set up the project environment

### Key Dependencies
- **clifpy**: Official Python implementation for working with CLIF data
- **pandas, numpy, pyarrow**: Data manipulation and parquet file support
- **plotnine**: Grammar of graphics visualization (ggplot2 for Python)
- **scikit-learn**: Machine learning for longitudinal sedation phenotyping

**Preferred method using uv:**
```bash
uv pip install -r requirements.txt
```
Note: uv automatically creates virtual environments and manages dependencies. It generates required files like `uv.lock` for reproducible builds. For more details, see the [CLIF uv guide by Zewei Whiskey Liao](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-data-huddles/blob/main/notes/uv-and-conv-commits.md).

**Alternative method using python3:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Run code

Detailed instructions on the code workflow are provided in the [code directory](code/README.md)

## Example CLIF Projects
- [CLIF Adult Sepsis Events](https://github.com/08wparker/CLIF_sepsis) - R implementation
- [CLIF Eligibility for Mobilization](https://github.com/kaveriC/CLIF-eligibility-for-mobilization) - Python implementation
- [CLIF Variation in Ventilation](https://github.com/ingra107/clif_vent_variation)
