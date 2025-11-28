# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CLIF (Common Longitudinal ICU data Format) research project studying overuse of sedation and analgesia in mechanically ventilated adult ICU patients. The project identifies phenotypes of deep sedation (RASS < -3 with continuous sedative/analgesic infusion) and applies unsupervised ML for longitudinal sedation pattern analysis.

**Key clinical definitions:**
- **Primary outcome**: Deep sedation = continuous infusion of sedative/analgesic + RASS < -3
- **Target population**: Adults (≥18 years) on invasive mechanical ventilation for >24 hours
- **Sedative medications tracked**: dexmedetomidine, fentanyl, hydromorphone, ketamine, lorazepam, midazolam, morphine, pentobarbital, propofol, remifentanil (see README.md table for full details)
- **Indications for deep sedation**: neuromuscular blockade, status epilepticus, elevated ICP, therapeutic hypothermia, organ donation

## Environment Setup

This project uses Python with optional R scripts. Set up using one of these methods:

**Using uv (preferred):**
```bash
uv init project-name
cd project-name
```

**Using python venv:**
```bash
python3 -m venv .mobilization
source .mobilization/bin/activate
pip install -r requirements.txt
```

Note: No requirements.txt or pyproject.toml exists yet - you'll need to create this based on project dependencies (pandas, numpy, pyarrow, clifpy, etc.).

## Configuration Architecture

**Critical**: The project uses a site-specific configuration system:

1. `config/config_template.json` - Template with placeholders
2. `config/config.json` - Site-specific settings (gitignored, never commit)
3. `utils/config.py` - Configuration loader used by all scripts

**Required config fields:**
- `site_name`: Institution identifier
- `tables_path`: Path to CLIF tables directory
- `file_type`: Data format ("csv", "parquet", or "fst")

All scripts import config via: `from utils import config` and access as `config['site_name']`, `config['tables_path']`, etc.

**Before running any code**: Ensure `config/config.json` exists with actual paths to CLIF data.

## Data Workflow

The project follows a three-stage pipeline:

### 1. Cohort Identification (`code/templates/Python/01_cohort_identification_template.py`)
- Apply inclusion/exclusion criteria
- Filter CLIF tables to cohort hospitalizations
- **Outputs**: `cohort_ids`, `cohort_data`, `cohort_summary`

### 2. Quality Control (`02_project_quality_checks_template.py`, `03_outlier_handling_template.py`)
- Project-specific QC checks
- Outlier handling using thresholds (formerly in `outlier-thresholds/` directory, now deleted)
- **Input**: `cohort_data`
- **Output**: `cleaned_cohort_data`

### 3. Analysis (`04_project_analysis_template.py`)
- Main statistical analysis and ML phenotyping
- **Input**: `cleaned_cohort_data`
- **Output**: Results saved to `output/final/`

## CLIF Data Format

**Required CLIF v2.1 tables** (see README.md for rationale):
- `clif_patient`: demographics (patient_id, race, ethnicity, sex)
- `clif_hospitalization`: admission data (hospitalization_id, admission_dttm, age_at_admission)
- `clif_vitals`: vital signs including heart rate, respiratory rate, blood pressure, SpO2
- `clif_labs`: laboratory results (especially lactate)
- `clif_medication_admin_continuous`: continuous medication infusions (sedatives, vasopressors, paralytics)
- `clif_respiratory_support`: ventilator settings and parameters
- `clif_adt`: admission/discharge/transfer events (used in cohort identification)

**Data reading pattern** (from `01_cohort_identification_template.py`):
```python
def read_data(filepath, filetype):
    # Handles CSV and Parquet formats
    # Prints load time and memory usage
    # Returns pandas DataFrame
```

All CLIF table filepaths follow pattern: `{tables_path}/clif_{table_name}.{file_type}`

## Output Conventions

All final results must be saved to `output/final/` with naming convention:
```
[RESULT_NAME]_[SITE_NAME]_[SYSTEM_TIME].[extension]
```

Example: `Table_One_2024-10-04_UCMC.pdf`

Use the config object to get site_name dynamically:
```python
from utils import config
from datetime import datetime
output_file = f"output/final/Table_One_{datetime.today().date()}_{config['site_name']}.pdf"
```

## Key Architecture Patterns

**Import structure in all scripts:**
```python
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import config
```

This allows scripts in `code/templates/Python/` to import from project root.

**Configuration access pattern:**
```python
from utils import config
site_name = config['site_name']
tables_path = config['tables_path']
file_type = config['file_type']
```

**Data filtering pattern for cohort:**
```python
clif_hospitalization['admission_dttm'] = pd.to_datetime(clif_hospitalization['admission_dttm'])
admissions_filtered = clif_hospitalization[
    (clif_hospitalization['admission_dttm'] >= start_date) &
    (clif_hospitalization['admission_dttm'] <= end_date)
]
cohort = admissions_filtered[admissions_filtered['age_at_admission'] >= 18]
```

## CLIF Resources

- [CLIF Data Dictionary](https://clif-consortium.github.io/website/data-dictionary.html)
- [ETL Tools](https://github.com/clif-consortium/CLIF/tree/main/etl-to-clif-resources)
- [clifpy Package](https://github.com/Common-Longitudinal-ICU-data-Format/clifpy) - Official Python implementation

## Related CLIF Projects

- [CLIF Adult Sepsis Events](https://github.com/08wparker/CLIF_sepsis) - R implementation
- [CLIF Eligibility for Mobilization](https://github.com/kaveriC/CLIF-eligibility-for-mobilization) - Python implementation
- [CLIF Variation in Ventilation](https://github.com/ingra107/clif_vent_variation)
