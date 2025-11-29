 ## Code directory

This directory contains scripts for the ICU oversedation project workflow. The workflow consists of three main steps: cohort identification, quality control, and analysis.

### Project Workflow

## 1. Cohort Identification (`01_cohort_identification.py`)

**Purpose**: Identify adult patients who received invasive mechanical ventilation (IMV) for ≥24 hours

**Inclusion criteria**:
- Adults (≥18 years old)
- Invasive mechanical ventilation for ≥24 hours

**How to run**:
```bash
cd /path/to/icu-oversedation
python3 code/01_cohort_identification.py
```

**Outputs** (saved to `output/`):
- `cohort_ids_{site_name}.csv` - List of hospitalization_id meeting criteria
- `cohort_summary_{site_name}.csv` - Summary with age, IMV timing, and duration
- `cohort_imv_times_{site_name}.csv` - Hospitalization IDs with first IMV timestamp
- `table_one_{site_name}.csv` - **Table One** with demographics, ventilator settings, vital signs, and SOFA scores
- `cohort_hospitalization_{site_name}.csv` - Filtered hospitalization table
- `cohort_respiratory_support_{site_name}.csv` - Filtered respiratory support table
- `cohort_vitals_{site_name}.csv` - Filtered vitals table
- `cohort_labs_{site_name}.csv` - Filtered labs table
- `cohort_medication_admin_continuous_{site_name}.csv` - Filtered medication table

**Configuration**:
Edit the following parameters in the script if needed:
- `start_date` / `end_date` - Optional date range filter
- `min_imv_hours` - Minimum IMV duration (default: 24 hours)
- IMV detection pattern (line 129-133) - Adjust keywords based on your data's `device_category` values

## 2. Quality Control (Coming Soon)

**Purpose**: Perform data quality checks and outlier handling

**Tasks**:
- Project-specific quality control checks on filtered cohort data
- Handle outliers in vitals, labs, and respiratory support data
- Clean and preprocess data for analysis

**Input**: Cohort data from step 1
**Output**: `cleaned_cohort_data`

## 3. Analysis (Coming Soon)

**Purpose**: Main statistical analysis and ML phenotyping

**Tasks**:
- Identify deep sedation episodes (RASS < -3 + continuous sedative/analgesic infusion)
- Characterize sedation patterns over time
- Unsupervised/self-supervised ML phenotyping of longitudinal sedation patterns
- Statistical analysis of risk factors

**Input**: Cleaned cohort data from step 2
**Output**: Results, figures, and tables saved to [`output/final/`](../output/README.md)

---

## Templates

Python templates are available in [`templates/Python/`](templates/Python/) for reference. 



