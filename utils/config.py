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



def load_table(table_name, data_folder="CLIF-MIMIC/Release 1.0.0 (MIMIC-IV 3.1 - CLIF 2.1.0)"):
    file_path = os.path.join(data_folder, table_name)
    return pd.read_parquet(file_path)

def standardize_dosage(df, )
    
if __name__ =="__main__":
    # Load the configuration
    config = load_config()