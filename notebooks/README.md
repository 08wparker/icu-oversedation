## Notebooks

This directory contains interactive marimo notebooks for exploring and analyzing the ICU oversedation cohort data.

### What is Marimo?

[Marimo](https://marimo.io) is a reactive Python notebook that automatically updates when you change your code or interact with UI elements. Unlike Jupyter, marimo notebooks are stored as pure Python files and can be run as scripts.

### Available Notebooks

#### `cohort_exploration.py`
Interactive exploration of the IMV cohort including:
- Cohort overview statistics
- Age and IMV duration distributions
- Medication administration patterns
- Vitals and respiratory support data

### Running Notebooks

**Important**: Always run marimo from the project root directory (not from within `notebooks/`)

**Interactive mode:**
```bash
# Make sure you're in the project root
cd /path/to/icu-oversedation

# Run the notebook
marimo edit notebooks/cohort_exploration.py
```
This opens the notebook in your browser at http://localhost:2718

**Run as script:**
```bash
marimo run notebooks/cohort_exploration.py
```

**Convert to static HTML:**
```bash
marimo export html notebooks/cohort_exploration.py > cohort_exploration.html
```

### Requirements

Make sure marimo is installed in your virtual environment:
```bash
uv pip install marimo
```
