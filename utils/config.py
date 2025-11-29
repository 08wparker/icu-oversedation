import json
import os
import pandas as pd

def load_config():
    json_path = os.path.join("config", "config.json")
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            config = json.load(file)
        print("Loaded configuration from config.json")
    else:
        raise FileNotFoundError("Configuration file not found.",
                                "Please create config.json based on the config_template.")
    
    return config
# Load the configuration
config = load_config()


def convert_weight_based_doses(med_df: pd.DataFrame, weights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert weight-based doses (e.g., mcg/kg/min, mg/kg/min) to absolute doses.

    Parameters:
        med_df (pd.DataFrame): Medication DataFrame with med_dose and med_dose_unit
        weights_df (pd.DataFrame): DataFrame with hospitalization_id and weight_kg
        
    Returns:
        pd.DataFrame: DataFrame with converted doses
    """
    print("\nConverting weight-based doses to absolute doses...")
    
    df = med_df.copy()
    
    # Merge with weights
    initial_len = len(df)
    df = df.merge(weights_df, on='hospitalization_id', how='left')
    
    # Count how many have weights
    n_with_weight = df['weight_kg'].notna().sum()
    print(f"Merged weights: {n_with_weight:,} out of {len(df):,} records have weight data ({n_with_weight/len(df)*100:.1f}%)")
    
    if 'med_dose_unit' not in df.columns:
        print("Warning: med_dose_unit column not found. Cannot convert weight-based doses.")
        return df
    
    # Create a copy of original dose and unit for reference
    df['med_dose_original'] = df['med_dose']
    df['med_dose_unit_original'] = df['med_dose_unit']
    
    # Identify weight-based units and convert
    weight_based_patterns = [
        (r'mcg/kg/min', 'mcg/min', 1),
        (r'mg/kg/min', 'mg/min', 1),
        (r'mcg/kg/hr', 'mcg/hr', 1/60),
        (r'mg/kg/hr', 'mg/hr', 1/60),
        (r'mcg/kg/h', 'mcg/hr', 1/60),
        (r'mg/kg/h', 'mg/hr', 1/60),
    ]
    
    # Convert doses where weight is available and unit is weight-based
    for pattern, target_unit, time_factor in weight_based_patterns:
        mask = (
            df['med_dose_unit'].str.contains(pattern, case=False, na=False) &
            df['weight_kg'].notna()
        )
        
        n_converted = mask.sum()
        if n_converted > 0:
            # Convert: dose * weight_kg (and adjust for time unit if needed)
            df.loc[mask, 'med_dose'] = df.loc[mask, 'med_dose'] * df.loc[mask, 'weight_kg'] * time_factor
            df.loc[mask, 'med_dose_unit'] = target_unit
            print(f"Converted {n_converted:,} doses from {pattern} to {target_unit}")
    
    # Show summary of units
    print("\nFinal dose unit distribution:")
    print(df['med_dose_unit'].value_counts())
    
    return df
