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

* Both continuous infusions from `med_admin_continuous` and bolus doses from `med_admin_intermittent`

| med_category     | description                                               | med_name_examples                                                                                     | med_group |
|------------------|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------|-----------|
| dexmedetomidine  | Alpha-2 agonist sedative                                  | DEXMEDETOMIDINE 200 MCG/50 ML IV INFUSION, DEXMEDETOMIDINE 0.2-1.5 MCG/KG/HR IV                       | sedation  |
| fentanyl         | Opioid analgesic                                          | FENTANYL 2500 MCG/50 ML IV INFUSION, FENTANYL 25-200 MCG/HR IV                                        | sedation  |
| hydromorphone    | Opioid analgesic                                          | HYDROMORPHONE 50 MG/50 ML IV INFUSION, HYDROMORPHONE 0.5-3 MG/HR IV PCA                               | sedation  |
| ketamine         | Dissociative anesthetic for sedation and analgesia        | KETAMINE 500 MG/50 ML IV INFUSION, KETAMINE 0.1-0.5 MG/KG/HR IV                                       | sedation  |
| lorazepam        | Benzodiazepine for anxiety and sedation                   | LORAZEPAM 50 MG/50 ML IV INFUSION, LORAZEPAM 1-4 MG/HR IV                                             | sedation  |
| midazolam        | Benzodiazepine for anxiety and sedation                   | MIDAZOLAM 100 MG/100 ML IV INFUSION, MIDAZOLAM 1-10 MG/HR IV                                          | sedation  |
| morphine         | Opiate analgesic                                          | MORPHINE 100 MG/100 ML IV INFUSION, MORPHINE 1-5 MG/HR IV PCA                                         | sedation  |
| pentobarbital    | Barbiturate for refractory intracranial hypertension      | PENTOBARBITAL 500 MG/250 ML IV INFUSION, PENTOBARBITAL 1-5 MG/KG/HR IV                                | sedation  |
| propofol         | Sedative-hypnotic for anesthesia and sedation             | PROPOFOL 1000 MG/100 ML IV INFUSION, PROPOFOL 5-50 MCG/KG/MIN IV                                      | sedation  |
| remifentanil     | Ultra-short acting opioid analgesic                       | REMIFENTANIL 5 MG/100 ML IV INFUSION, REMIFENTANIL 0.05-2 MCG/KG/MIN IV                               | sedation  |



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
- **marimo**: Reactive Python notebooks for interactive data exploration

**Preferred method using uv:**
```bash
# Create virtual environment
uv venv

# Activate the environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```
Note: uv automatically manages dependencies and generates required files like `uv.lock` for reproducible builds. For more details, see the [CLIF uv guide by Zewei Whiskey Liao](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-data-huddles/blob/main/notes/uv-and-conv-commits.md).

**Alternative method using python3:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Run code

Detailed instructions on the code workflow are provided in the [code directory](code/README.md)

## 4. Interactive exploration with marimo notebooks

Explore your cohort data interactively using marimo notebooks:
```bash
marimo edit notebooks/cohort_exploration.py
```

See the [notebooks directory](notebooks/README.md) for more details on available notebooks and how to use marimo.

## Example CLIF Projects
- [CLIF Adult Sepsis Events](https://github.com/08wparker/CLIF_sepsis) - R implementation
- [CLIF Eligibility for Mobilization](https://github.com/kaveriC/CLIF-eligibility-for-mobilization) - Python implementation
- [CLIF Variation in Ventilation](https://github.com/ingra107/clif_vent_variation)
