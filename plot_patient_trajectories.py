"""
Plot patient medication trajectories aligned to start time (hour 0).
Each patient starts at hour 0 when they first receive the medication,
plotted for 72 hours (3 days) with day markers.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pyarrow.parquet as pq
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap


def create_period_indicator(df: pd.DataFrame, 
                            date_col: str = 'admin_dttm',
                            group_by: list = None,
                            threshold_hours: float = 24.0) -> pd.DataFrame:
    """
    Create a period indicator column where:
    - First date is 0
    - When date changes (beyond threshold), increment to 1, 2, 3, etc.
    - Resets for each group (e.g., per patient, per medication)
    
    Parameters:
        df (pd.DataFrame): DataFrame with date column
        date_col (str): Name of the date/datetime column (default: 'admin_dttm')
        group_by (list): Columns to group by (default: ['hospitalization_id'])
        threshold_hours (float): Hours threshold for considering it a new period (default: 24.0)
        
    Returns:
        pd.DataFrame: DataFrame with 'period_indicator' column (0, 1, 2, ...)
    """
    df = df.copy()
    
    if group_by is None:
        group_by = ['hospitalization_id']
    
    if date_col not in df.columns:
        print(f"Warning: {date_col} column not found. Skipping period indicator creation.")
        return df
    
    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Sort by grouping columns and date
    df = df.sort_values(group_by + [date_col])
    
    def assign_periods(group):
        """Assign period indicators within a group"""
        group = group.sort_values(date_col).copy()
        periods = np.zeros(len(group), dtype=int)
        
        if len(group) == 0:
            return pd.Series(periods, index=group.index)
        
        # First record is always period 0
        periods[0] = 0
        
        if len(group) == 1:
            return pd.Series(periods, index=group.index)
        
        # Check for date changes
        dates = group[date_col].values
        current_period = 0
        
        for i in range(1, len(dates)):
            time_diff = (dates[i] - dates[i-1]).total_seconds() / 3600.0
            
            # If time gap exceeds threshold, increment period
            if time_diff > threshold_hours or pd.isna(dates[i-1]):
                current_period += 1
            
            periods[i] = current_period
        
        return pd.Series(periods, index=group.index)
    
    # Apply to each group
    df['period_indicator'] = df.groupby(group_by, group_keys=False).apply(assign_periods).values
    
    n_periods = df['period_indicator'].nunique()
    print(f"Created period indicators: {n_periods} unique periods across all groups")
    
    return df


def load_medication_data(meds_parquet_file: str, 
                        medication: str = 'propofol',
                        hospitalization_id: int = None,
                        n_random_patients: int = None) -> pd.DataFrame:
    """
    Load medication data from parquet file and filter for specific medication and patient(s).
    
    Parameters:
        meds_parquet_file (str): Path to medication parquet file
        medication (str): Medication category to filter (default: 'propofol')
        hospitalization_id (int): Specific hospitalization_id to filter (optional)
        n_random_patients (int): Number of random patients to select (optional)
        
    Returns:
        pd.DataFrame: Filtered medication data
    """
    print(f"\nLoading medication data from: {meds_parquet_file}")
    
    # Load parquet file
    table = pq.read_table(meds_parquet_file)
    df = table.to_pandas()
    
    print(f"Total rows loaded: {len(df):,}")
    
    # Filter for specific medication
    if 'med_category' in df.columns:
        df = df[df['med_category'] == medication].copy()
        print(f"Filtered to {medication}: {len(df):,} rows")
    else:
        print("Warning: med_category column not found")
        return pd.DataFrame()
    
    # Filter for specific patient if provided
    if hospitalization_id is not None:
        df = df[df['hospitalization_id'] == hospitalization_id].copy()
        print(f"Filtered to hospitalization_id {hospitalization_id}: {len(df):,} rows")
    elif n_random_patients is not None:
        # Select random patients
        unique_patients = df['hospitalization_id'].unique()
        n_available = len(unique_patients)
        
        if n_random_patients > n_available:
            print(f"Warning: Only {n_available} patients available, using all of them")
            selected_patients = unique_patients
        else:
            selected_patients = np.random.choice(unique_patients, size=n_random_patients, replace=False)
            print(f"Randomly selected {n_random_patients} patients from {n_available} available")
        
        df = df[df['hospitalization_id'].isin(selected_patients)].copy()
        print(f"Filtered to selected patients: {len(df):,} rows")
        print(f"Selected patient IDs: {sorted(selected_patients)}")
    
    # Convert admin_dttm to datetime if needed
    if 'admin_dttm' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['admin_dttm']):
            df['admin_dttm'] = pd.to_datetime(df['admin_dttm'], errors='coerce')
    
    # Sort by time
    df = df.sort_values(['hospitalization_id', 'admin_dttm'])
    
    return df


def filter_to_time_window(df: pd.DataFrame, 
                          max_hours: float = 72.0) -> pd.DataFrame:
    """
    Filter data to show 72 hours from each patient's first dose.
    Keeps real calendar time (doesn't align to hour 0).
    
    Parameters:
        df (pd.DataFrame): Medication DataFrame with admin_dttm
        max_hours (float): Maximum hours to plot from first dose (default: 72.0)
        
    Returns:
        pd.DataFrame: Filtered DataFrame with data within max_hours of first dose
    """
    df = df.copy()
    
    if 'admin_dttm' not in df.columns:
        print("Error: admin_dttm column not found")
        return df
    
    # Group by hospitalization_id and find first dose time for each patient
    first_dose_times = df.groupby('hospitalization_id')['admin_dttm'].transform('min')
    
    # Calculate hours from first dose for filtering (but keep real time for plotting)
    hours_from_start = (df['admin_dttm'] - first_dose_times).dt.total_seconds() / 3600.0
    
    # Filter to only show data within max_hours
    initial_len = len(df)
    df = df[hours_from_start <= max_hours].copy()
    removed = initial_len - len(df)
    
    if removed > 0:
        print(f"Filtered to {max_hours} hours from first dose: removed {removed:,} rows beyond {max_hours} hours")
    
    return df


def plot_patient_trajectories(df: pd.DataFrame,
                              medication: str = 'propofol',
                              output_file: str = 'output/final/patient_trajectories.png',
                              patient_duration_hours: float = 48.0,
                              x_axis_max_hours: float = 72.0) -> None:
    """
    Plot patient trajectories using TIME only (0-72 hours on x-axis).
    Each patient starts at their ACTUAL TIME on the x-axis (e.g., 20:17, 01:00, 05:00).
    Each patient shows their 48-hour trajectory from their start time.
    X-axis shows TIME (hours:minutes), not calendar dates.
    Day markers show Day 0, Day 1, Day 2 boundaries at 00:00, 24:00, 48:00.
    
    Parameters:
        df (pd.DataFrame): Medication DataFrame with admin_dttm and med_dose
        medication (str): Medication name for title
        output_file (str): Path to save the plot
        patient_duration_hours (float): Hours to show per patient (default: 48.0)
        x_axis_max_hours (float): Maximum hours on x-axis (default: 72.0)
    """
    print(f"\nCreating patient trajectory plot for {medication}...")
    
    if len(df) == 0:
        print("No data to plot")
        return
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get unique patients
    unique_patients = df['hospitalization_id'].unique()
    n_patients = len(unique_patients)
    print(f"Plotting {n_patients} patient(s)")
    
    # Get first dose time for each patient
    first_dose_times = df.groupby('hospitalization_id')['admin_dttm'].min()
    
    # Calculate each patient's start position on x-axis based on their actual time of day
    patient_start_positions = {}
    for patient_id, first_dose in first_dose_times.items():
        # Convert to hours:minutes as decimal hours (e.g., 20:17 = 20.283 hours)
        start_hours = first_dose.hour + first_dose.minute / 60.0 + first_dose.second / 3600.0
        patient_start_positions[patient_id] = start_hours
    
    print(f"\nPatient start positions on x-axis:")
    for patient_id in sorted(unique_patients):
        start_pos = patient_start_positions[patient_id]
        start_time = first_dose_times[patient_id]
        print(f"  Patient {patient_id}: starts at {start_pos:.2f} hours ({start_time.hour:02d}:{start_time.minute:02d})")
    
    # Collect all x and y positions for density plot
    all_x_positions = []
    all_doses = []
    patient_data_list = []  # Store for step plots later
    
    # First pass: collect all data for density calculation
    for patient_id in unique_patients:
        patient_df = df[df['hospitalization_id'] == patient_id].copy()
        patient_df = patient_df.sort_values('admin_dttm')
        
        start_time = first_dose_times[patient_id]
        hours_from_start = (patient_df['admin_dttm'] - start_time).dt.total_seconds() / 3600.0
        
        mask = hours_from_start <= patient_duration_hours
        patient_df = patient_df[mask].copy()
        hours_from_start = hours_from_start[mask]
        
        # Handle duplicates
        patient_df['hours_from_start'] = hours_from_start.values
        patient_df = patient_df.sort_values(['hours_from_start', 'med_dose'], ascending=[True, False])
        patient_df = patient_df.drop_duplicates(subset=['hours_from_start'], keep='first')
        patient_df = patient_df.sort_values('hours_from_start')
        
        hours_from_start = patient_df['hours_from_start'].values
        doses = patient_df['med_dose'].values
        
        start_pos = patient_start_positions[patient_id]
        x_positions = start_pos + hours_from_start
        
        mask_x = x_positions <= x_axis_max_hours
        x_positions = x_positions[mask_x]
        doses = doses[mask_x]
        
        # Store for density plot
        all_x_positions.extend(x_positions)
        all_doses.extend(doses)
        
        # Store for step plot
        patient_data_list.append({
            'patient_id': patient_id,
            'x_positions': x_positions,
            'doses': doses,
            'start_time': start_time
        })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create 2D density plot (hexbin) as background
    if len(all_x_positions) > 0 and len(all_doses) > 0:
        hb = ax.hexbin(all_x_positions, all_doses, 
                       gridsize=50, 
                       cmap='Blues', 
                       alpha=0.3,
                       mincnt=1,
                       linewidths=0.1)
        # Add colorbar for density
        cb = plt.colorbar(hb, ax=ax, label='Density (number of patients)')
    
    # Plot each patient step plot on top of density
    for patient_data in patient_data_list:
        patient_id = patient_data['patient_id']
        x_positions = patient_data['x_positions']
        doses = patient_data['doses']
        start_time = patient_data['start_time']
        
        if len(x_positions) == 0:
            continue
        
        # Get the time of day they started (for label)
        start_hour_min = f"{start_time.hour:02d}:{start_time.minute:02d}"
        
        # Plot step plot (horizontal steps, vertical jumps at data points) - on top of density
        # ax.step(x_positions, 
        #         doses,
        #         where='post',  # Steps happen after the point (right edge)
        #         marker='o', 
        #         markersize=2,
        #         linewidth=1.0,
        #         alpha=0.6,
        #         label=f'Patient {patient_id} (started at {start_hour_min})',
        #         color='red' if n_patients <= 5 else None)  # Highlight individual lines if few patients
    
    # Add day markers at 00:00, 24:00, 48:00 (hours)
    for day in range(3):  # Day 0, 1, 2
        hour = day * 24
        if hour <= x_axis_max_hours:
            ax.axvline(x=hour, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.text(hour, ax.get_ylim()[1] * 0.95, f'Day {day}\n{hour:02d}:00', 
                   ha='center', va='top', fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Format x-axis as hours (0-72)
    ax.set_xlabel('Time (Hours: 0:00 to 72:00)', fontsize=12)
    ax.set_ylabel('Medication Dose (raw med_dose)', fontsize=12)
    ax.set_title(f'{medication.title()} - Patient Trajectories\n'
                 f'X-axis shows TIME. Each patient starts at their actual time and shows {patient_duration_hours} hours. Total: {n_patients} patient(s)',
                 fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, x_axis_max_hours)
    ax.set_ylim(0, 200)
    
    # Set x-axis ticks every 6 hours with time format
    ax.set_xticks(range(0, int(x_axis_max_hours) + 1, 6))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, int(x_axis_max_hours) + 1, 6)])
    
    # Legend
    if n_patients <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        print(f"  (Skipping legend for {n_patients} patients to avoid clutter)")
    
    plt.tight_layout()
    
    # Save
    print(f"Saving plot to: {output_file}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved successfully!")
    
    plt.close()


def main():
    """
    Main function to plot patient trajectories for propofol.
    Starts with one hospitalization_id patient.
    """
    # File paths
    meds_parquet_file = "code/Release 1.0.0 (MIMIC-IV 3.1 - CLIF 2.1.0)/clif_medication_admin_continuous.parquet"
    
    # Configuration
    medication = 'propofol'
    n_random_patients = 1000  # Plot 3 random patients
    patient_duration_hours = 48.0  # Each patient shows 48 hours of data
    x_axis_max_hours = 72.0  # X-axis spans 0-72 hours to accommodate all patients
    
    try:
        # Load medication data - select 3 random patients
        df = load_medication_data(
            meds_parquet_file,
            medication=medication,
            n_random_patients=n_random_patients
        )
        
        if len(df) == 0:
            print("No data found. Please check medication name.")
            return
        
        # Get unique patients
        unique_patients = df['hospitalization_id'].unique()
        n_patients = len(unique_patients)
        
        print(f"\nPatient data summary:")
        print(f"  Number of patients: {n_patients}")
        print(f"  Total records: {len(df):,}")
        
        # Show each patient's actual start time (real time, not aligned)
        print(f"\nPatient start times (real calendar time):")
        for patient_id in sorted(unique_patients):
            patient_df = df[df['hospitalization_id'] == patient_id]
            first_dose_time = patient_df['admin_dttm'].min()
            print(f"  Patient {patient_id}: First dose at {first_dose_time} "
                  f"(Day {first_dose_time.day}, {first_dose_time.hour:02d}:{first_dose_time.minute:02d})")
        
        print(f"\n  Dose range: {df['med_dose'].min():.2f} to {df['med_dose'].max():.2f}")
        if 'med_dose_unit' in df.columns:
            print(f"  Units: {df['med_dose_unit'].unique()}")
        
        # Filter to patient_duration_hours from each patient's first dose
        df = filter_to_time_window(df, max_hours=patient_duration_hours)
        
        print(f"\nAfter filtering to {patient_duration_hours} hours from first dose:")
        print(f"  Date range: {df['admin_dttm'].min()} to {df['admin_dttm'].max()}")
        
        # Plot - each patient starts at their actual time on x-axis
        output_file = f"output/final/patient_trajectories_{medication}_{n_patients}patients_48hours.png"
        plot_patient_trajectories(
            df, 
            medication=medication, 
            output_file=output_file, 
            patient_duration_hours=patient_duration_hours,
            x_axis_max_hours=x_axis_max_hours
        )
        
        print("\n" + "="*50)
        print("Plotting completed successfully!")
        print("="*50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

