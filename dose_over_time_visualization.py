import pandas as pd
import numpy as np
from datetime import datetime
import pyarrow.parquet as pq
from pathlib import Path
from plotnine import (
    ggplot, aes, geom_line, geom_point, facet_wrap, geom_segment, geom_rect,
    labs, theme_minimal, theme, element_text, scale_x_datetime,
    scale_color_discrete, scale_y_continuous, ggsave, geom_step,
    geom_boxplot, geom_jitter
)


def load_cohort_hospitalization_ids(cohort_filepath: str) -> list:
    """
    Load hospitalization_ids from cohort CSV file.
    
    Parameters:
        cohort_filepath (str): Path to the cohort CSV file
        
    Returns:
        list: List of hospitalization_id values
    """
    print(f"\nLoading cohort hospitalization IDs from: {cohort_filepath}")
    
    cohort_df = pd.read_csv(cohort_filepath)
    
    if 'hospitalization_id' not in cohort_df.columns:
        raise ValueError(f"hospitalization_id column not found in cohort file. Available columns: {cohort_df.columns.tolist()}")
    
    cohort_ids = cohort_df['hospitalization_id'].dropna().unique().tolist()
    
    print(f"Loaded {len(cohort_ids):,} unique hospitalization_ids from cohort file")
    print(f"Cohort file contains {len(cohort_df):,} total rows")
    
    return cohort_ids


def read_parquet_file(filepath: str) -> pd.DataFrame:

    print(f"Loading data from: {filepath}")
    start_time = datetime.now()
    
    table = pq.read_table(filepath)
    df = table.to_pandas()
    
    load_time = (datetime.now() - start_time).total_seconds()
    dataset_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    print(f"Time taken to load: {load_time:.2f} seconds")
    print(f"Dataset size: {dataset_size_mb:.2f} MB")
    print(f"Number of rows: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}\n")
    
    return df


def load_patient_weights(vitals_filepath: str, hospitalization_ids: list = None) -> pd.DataFrame:
    """
    Load patient weights from vitals table.
    
    Parameters:
        vitals_filepath (str): Path to the vitals parquet file
        hospitalization_ids (list): Optional list of hospitalization_ids to filter
        
    Returns:
        pd.DataFrame: DataFrame with hospitalization_id and weight_kg
    """
    print(f"\nLoading patient weights from: {vitals_filepath}")
    
    # Load vitals data
    table = pq.read_table(vitals_filepath)
    vitals_df = table.to_pandas()
    
    # Filter for weight data
    if 'vital_category' in vitals_df.columns:
        weight_df = vitals_df[vitals_df['vital_category'] == 'weight_kg'].copy()
        print(f"Found {len(weight_df):,} weight records")
    else:
        raise ValueError("vital_category column not found in vitals table")
    
    # Filter by hospitalization_ids if provided
    if hospitalization_ids is not None:
        weight_df = weight_df[weight_df['hospitalization_id'].isin(hospitalization_ids)].copy()
        print(f"Filtered to {len(weight_df):,} weight records for specified hospitalizations")
    
    # Get the most recent weight per hospitalization (or average if multiple)
    # Convert recorded_dttm to datetime if needed
    if 'recorded_dttm' in weight_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(weight_df['recorded_dttm']):
            weight_df['recorded_dttm'] = pd.to_datetime(weight_df['recorded_dttm'], errors='coerce')
        
        # Get the most recent weight per hospitalization_id
        weight_df = weight_df.sort_values('recorded_dttm')
        weight_df = weight_df.groupby('hospitalization_id').last().reset_index()
    else:
        # If no timestamp, just take the average weight per hospitalization
        weight_df = weight_df.groupby('hospitalization_id').agg({
            'vital_value': 'mean'
        }).reset_index()
    
    # Rename vital_value to weight_kg
    if 'vital_value' in weight_df.columns:
        weight_df = weight_df.rename(columns={'vital_value': 'weight_kg'})
    
    # Select only necessary columns
    weight_df = weight_df[['hospitalization_id', 'weight_kg']].copy()
    
    # Remove rows with missing or invalid weights
    initial_len = len(weight_df)
    weight_df = weight_df[weight_df['weight_kg'].notna() & (weight_df['weight_kg'] > 0)].copy()
    if len(weight_df) < initial_len:
        print(f"Removed {initial_len - len(weight_df)} rows with invalid weights")
    
    print(f"Loaded weights for {len(weight_df):,} hospitalizations")
    print(f"Weight range: {weight_df['weight_kg'].min():.2f} - {weight_df['weight_kg'].max():.2f} kg")
    
    return weight_df


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


def filter_sedation_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame for sedation medications.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    print("Filtering for sedation medications...")
    print(f"Available columns: {df.columns.tolist()}")
    
    # Check which column exists for filtering - prioritize med_group
    if 'med_group' in df.columns:
        # Check unique values in med_group to see if 'sedation' exists
        unique_groups = df['med_group'].unique()
        print(f"Unique values in med_group: {unique_groups}")
        
        if 'sedation' in unique_groups:
            sedation_df = df[df['med_group'] == 'sedation'].copy()
            print(f"Found {len(sedation_df):,} rows with med_group == 'sedation'")
        else:
            print(f"Warning: 'sedation' not found in med_group values. Available values: {unique_groups}")
            # Try case-insensitive match
            sedation_df = df[df['med_group'].str.lower() == 'sedation'].copy()
            if len(sedation_df) > 0:
                print(f"Found {len(sedation_df):,} rows with med_group (case-insensitive) == 'sedation'")
            else:
                raise ValueError(f"'sedation' not found in med_group. Available values: {unique_groups}")
    elif 'med_category' in df.columns:
        # If med_group doesn't exist, try filtering by common sedation med categories
        sedation_categories = ['hydromorphone', 'fentanyl', 'lorazepam', 'midazolam', 'propofol', 
                              'dexmedetomidine', 'morphine']
        sedation_df = df[df['med_category'].isin(sedation_categories)].copy()
        print(f"med_group column not found. Filtering by med_category in: {sedation_categories}")
        print(f"Found {len(sedation_df):,} rows")
    else:
        raise ValueError("Neither 'med_group' nor 'med_category' column found in the data")
    
    if len(sedation_df) == 0:
        raise ValueError("No sedation data found. Please check the filtering criteria.")
    
    # Display summary statistics
    if 'med_category' in sedation_df.columns:
        print("\nMedication categories in filtered data:")
        print(sedation_df['med_category'].value_counts())
    
    if 'med_name' in sedation_df.columns:
        print("\nTop 10 medication names:")
        print(sedation_df['med_name'].value_counts().head(10))
    
    # Show date range
    if 'admin_dttm' in sedation_df.columns:
        print(f"\nDate range: {sedation_df['admin_dttm'].min()} to {sedation_df['admin_dttm'].max()}")
    
    # Show unique hospitalizations
    if 'hospitalization_id' in sedation_df.columns:
        n_unique_hosp = sedation_df['hospitalization_id'].nunique()
        print(f"Number of unique hospitalizations: {n_unique_hosp:,}")
    
    return sedation_df


def prepare_data_for_visualization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for visualization by converting datetime and handling missing values.
    Filters for start, stop, and dose_change events in mar_action_category.
    """
    print("\nPreparing data for visualization...")
    
    df = df.copy()
    
    # Don't filter by mar_action_category - include all events
    # (This ensures we capture all dose changes, not just specific action types)
    if 'mar_action_category' in df.columns:
        print(f"\nDistribution of mar_action_category (keeping all):")
        print(df['mar_action_category'].value_counts())
        print(f"Total rows before any filtering: {len(df):,}")
    else:
        print("Warning: mar_action_category column not found. Using all rows.")
    
    # Convert admin_dttm to datetime if it's not already
    if 'admin_dttm' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['admin_dttm']):
            df['admin_dttm'] = pd.to_datetime(df['admin_dttm'], errors='coerce')
        # Remove rows with invalid dates
        initial_len = len(df)
        df = df.dropna(subset=['admin_dttm'])
        if len(df) < initial_len:
            print(f"Removed {initial_len - len(df)} rows with invalid dates")
    
    # Handle missing doses - keep all doses including zero (they might represent stops)
    if 'med_dose' in df.columns:
        initial_len = len(df)
        df = df[df['med_dose'].notna()].copy()
        if len(df) < initial_len:
            print(f"Removed {initial_len - len(df)} rows with missing doses")
    
    # Sort by datetime
    if 'admin_dttm' in df.columns:
        df = df.sort_values(['hospitalization_id', 'admin_dttm'])
    
    print(f"Final dataset: {len(df):,} rows")
    
    # Show patient count
    if 'hospitalization_id' in df.columns:
        n_patients = df['hospitalization_id'].nunique()
        print(f"Unique patients (hospitalizations): {n_patients:,}")
    
    return df


def create_infusion_periods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create continuous infusion periods from start/stop events using mar_action_category.
    Uses admin_dttm to calculate durations between start and stop events.
    
    Parameters:
        df (pd.DataFrame): Prepared DataFrame with sedation data
        
    Returns:
        pd.DataFrame: DataFrame with infusion periods (start_dttm, end_dttm, duration_hours, dose)
    """
    print("\nCreating continuous infusion periods from start/stop events...")
    
    if 'mar_action_category' not in df.columns:
        print("Warning: mar_action_category not found. Using all events as points.")
        return df
    
    # Check available mar_action_category values
    unique_actions = df['mar_action_category'].dropna().unique()
    print(f"Available mar_action_category values: {unique_actions}")
    
    # Filter for start and stop events (case-insensitive)
    start_mask = df['mar_action_category'].str.lower().str.contains('start|going', case=False, na=False)
    stop_mask = df['mar_action_category'].str.lower().str.contains('stop|stopped|held|paused', case=False, na=False)
    
    start_events = df[start_mask].copy()
    stop_events = df[stop_mask].copy()
    
    print(f"Found {len(start_events):,} start events")
    print(f"Found {len(stop_events):,} stop events")
    
    if len(start_events) == 0:
        print("Warning: No start events found. Using all data points.")
        return df
    
    # Identify grouping columns for matching start/stop pairs
    grouping_cols = ['hospitalization_id']
    if 'med_order_id' in df.columns:
        grouping_cols.append('med_order_id')
    elif 'med_name' in df.columns:
        grouping_cols.append('med_name')
    elif 'med_category' in df.columns:
        grouping_cols.append('med_category')
    
    # Create periods by matching start and stop events
    periods = []
    
    for name, group_df in df.groupby(grouping_cols):
        if len(grouping_cols) == 1:
            group_dict = {grouping_cols[0]: name}
        else:
            group_dict = dict(zip(grouping_cols, name if isinstance(name, tuple) else [name]))
        
        group_starts = start_events.copy()
        group_stops = stop_events.copy()
        
        for col, val in group_dict.items():
            group_starts = group_starts[group_starts[col] == val]
            group_stops = group_stops[group_stops[col] == val]
        
        group_starts = group_starts.sort_values('admin_dttm')
        group_stops = group_stops.sort_values('admin_dttm')
        
        # Match starts with next stop
        for idx, start_row in group_starts.iterrows():
            start_time = start_row['admin_dttm']
            start_dose = start_row.get('med_dose', 0)
            
            # Find the next stop after this start
            next_stops = group_stops[group_stops['admin_dttm'] > start_time]
            
            if len(next_stops) > 0:
                end_time = next_stops.iloc[0]['admin_dttm']
            else:
                # No stop found - use last event time in this group
                end_time = group_df['admin_dttm'].max()
            
            duration_hours = (end_time - start_time).total_seconds() / 3600.0
            
            if duration_hours <= 0:
                continue  # Skip invalid periods
            
            # Create period record
            period_dict = group_dict.copy()
            period_dict.update({
                'start_dttm': start_time,
                'end_dttm': end_time,
                'duration_hours': duration_hours,
                'med_dose': start_dose,
                'med_dose_unit': start_row.get('med_dose_unit', 'unknown'),
                'med_name': start_row.get('med_name', ''),
                'med_category': start_row.get('med_category', '')
            })
            periods.append(period_dict)
    
    if len(periods) == 0:
        print("Warning: No periods created. Returning original data.")
        return df
    
    periods_df = pd.DataFrame(periods)
    
    print(f"Created {len(periods_df):,} infusion periods")
    print(f"Total duration: {periods_df['duration_hours'].sum():.2f} hours")
    print(f"Average period duration: {periods_df['duration_hours'].mean():.2f} hours")
    print(f"Median period duration: {periods_df['duration_hours'].median():.2f} hours")
    
    return periods_df


def create_dose_over_time_plot(df: pd.DataFrame, 
                                output_path: str = "output/final/sedation_dose_over_time.png",
                                by_medication: bool = True,
                                sample_size: int = None) -> None:
    """
    Create a simple line graph showing dose over time for sedation medications.
    Uses admin_dttm for time and med_dose for dose values.
    Handles start, stop, and dose_change events.
    
    Parameters:
        df (pd.DataFrame): Prepared DataFrame with sedation data
        output_path (str): Path to save the plot
        by_medication (bool): If True, create separate plots for each medication
        sample_size (int): If specified, randomly sample this many hospitalizations
    """
    print(f"\nCreating dose over time line graph...")
    
    # Sample hospitalizations if specified
    if sample_size is not None and sample_size > 0:
        unique_hosp_ids = df['hospitalization_id'].unique()
        if len(unique_hosp_ids) > sample_size:
            sampled_ids = np.random.choice(unique_hosp_ids, size=sample_size, replace=False)
            df = df[df['hospitalization_id'].isin(sampled_ids)].copy()
            print(f"Sampled {sample_size} hospitalizations for visualization (from {len(unique_hosp_ids)} total)")
        else:
            print(f"Using all {len(unique_hosp_ids)} hospitalizations")
    else:
        unique_hosp_ids = df['hospitalization_id'].unique()
        print(f"Using all {len(unique_hosp_ids)} hospitalizations for visualization")
    
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine grouping column
    group_col = 'med_category' if 'med_category' in df.columns else 'med_name'
    
    # Create line graph visualization
    if by_medication and group_col in df.columns:
        print(f"Creating faceted line graph by {group_col}...")
        plot = (
            ggplot(df, aes(x='admin_dttm', y='med_dose', color='hospitalization_id', group='hospitalization_id'))
            + geom_line(alpha=0.7, size=0.8)
            + geom_point(alpha=0.6, size=1.5)
            + facet_wrap(f'~ {group_col}', scales='free_y', ncol=2)
            + labs(
                title='Sedation Dose Over Time by Medication',
                subtitle='Weight-based doses converted to absolute doses using hospitalization_id',
                x='Time (admin_dttm)',
                y='Medication Dose (converted to absolute units)',
                color='Hospitalization ID'
            )
            + theme_minimal()
            + theme(
                axis_text_x=element_text(angle=45, hjust=1, size=8),
                axis_text_y=element_text(size=8),
                strip_text=element_text(size=10),
                plot_title=element_text(size=14, face='bold'),
                plot_subtitle=element_text(size=10),
                legend_position='none'
            )
            + scale_x_datetime(date_labels='%Y-%m-%d %H:%M')
        )
    else:
        print("Creating single line graph...")
        plot = (
            ggplot(df, aes(x='admin_dttm', y='med_dose', color='hospitalization_id', group='hospitalization_id'))
            + geom_line(alpha=0.7, size=0.8)
            + geom_point(alpha=0.6, size=1.5)
            + labs(
                title='Sedation Dose Over Time',
                subtitle='Weight-based doses converted to absolute doses using hospitalization_id',
                x='Time (admin_dttm)',
                y='Medication Dose (converted to absolute units)',
                color='Hospitalization ID'
            )
            + theme_minimal()
            + theme(
                axis_text_x=element_text(angle=45, hjust=1, size=8),
                axis_text_y=element_text(size=8),
                plot_title=element_text(size=14, face='bold'),
                plot_subtitle=element_text(size=10),
                legend_position='none'
            )
            + scale_x_datetime(date_labels='%Y-%m-%d %H:%M')
        )
    
    # Save the plot
    print(f"Saving plot to: {output_path}")
    ggsave(plot, output_path, width=14, height=10, dpi=300)
    print("Plot saved successfully!")


def create_individual_sedative_plots(df: pd.DataFrame,
                                      output_dir: str = "output/final/",
                                      sample_size: int = None) -> None:
    """
    Create individual PNG plots for each sedative medication.
    Each plot shows individual patient trajectories - one line per hospitalization_id.
    Time on x-axis, dose on y-axis.
    
    Parameters:
        df (pd.DataFrame): Prepared DataFrame with sedation data
        output_dir (str): Directory to save the PNG plots
        sample_size (int): If specified, randomly sample this many hospitalizations
    """
    print(f"\nCreating individual PNG plots for each sedative medication...")
    
    # Sample hospitalizations if specified
    if sample_size is not None and sample_size > 0:
        unique_hosp_ids = df['hospitalization_id'].unique()
        if len(unique_hosp_ids) > sample_size:
            sampled_ids = np.random.choice(unique_hosp_ids, size=sample_size, replace=False)
            df = df[df['hospitalization_id'].isin(sampled_ids)].copy()
            print(f"Sampled {sample_size} hospitalizations for visualization (from {len(unique_hosp_ids)} total)")
        else:
            print(f"Using all {len(unique_hosp_ids)} hospitalizations")
    else:
        unique_hosp_ids = df['hospitalization_id'].unique()
        print(f"Using all {len(unique_hosp_ids)} hospitalizations for visualization")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine grouping column
    group_col = 'med_category' if 'med_category' in df.columns else 'med_name'
    
    # Get unique medications
    unique_meds = df[group_col].dropna().unique()
    print(f"Found {len(unique_meds)} medications: {unique_meds.tolist()}")
    
    # Create a separate plot for each medication
    for med in unique_meds:
        print(f"\nCreating plot for: {med}")
        med_df = df[df[group_col] == med].copy()
        med_df = med_df.sort_values(['hospitalization_id', 'admin_dttm'])
        
        # Count unique hospitalizations for this medication
        n_patients = med_df['hospitalization_id'].nunique()
        n_events = len(med_df)
        print(f"  - {n_patients} patients (hospitalizations) with {n_events:,} total dose events")
        
        # Calculate relative time (hours from first dose) for each patient
        # Each patient's timeline starts at 0 (their first dose time)
        med_df['time_hours'] = med_df.groupby('hospitalization_id')['admin_dttm'].transform(
            lambda x: (x - x.min()).dt.total_seconds() / 3600.0
        )
        
        # Always use hours for time unit
        time_col = 'time_hours'
        time_unit = 'Hours'
        max_time_hours = med_df['time_hours'].max()
        print(f"  - Time range: 0 to {max_time_hours:.1f} hours ({max_time_hours/24:.1f} days)")
        
        # Create filename (sanitize medication name)
        safe_med_name = str(med).replace('/', '_').replace('\\', '_').replace(' ', '_')
        output_file = output_path / f"sedation_dose_{safe_med_name}.png"
        
        # Create the plot with relative time (horizontal progression from time 0)
        plot = (
            ggplot(med_df, aes(x=time_col, y='med_dose', color='hospitalization_id', group='hospitalization_id'))
            + geom_line(alpha=0.7, size=0.8)
            + geom_point(alpha=0.6, size=2)
            + labs(
                title=f'{med} - Individual Patient Trajectories',
                subtitle=f'Each line represents one patient (hospitalization_id) starting from their first dose (time=0). Total: {n_patients} patients.',
                x=f'Time from First Dose ({time_unit})',
                y='Medication Dose (converted to absolute units)',
                color='Hospitalization ID'
            )
            + theme_minimal()
            + theme(
                axis_text_x=element_text(angle=0, hjust=0.5, size=9),
                axis_text_y=element_text(size=9),
                plot_title=element_text(size=14, face='bold'),
                plot_subtitle=element_text(size=10),
                legend_position='none'  # Hide legend to avoid clutter with many patients
            )
        )
        
        # Save the plot
        print(f"  - Saving to: {output_file}")
        ggsave(plot, str(output_file), width=12, height=8, dpi=300)
        print(f"  - ✓ Saved successfully!")
    
    print(f"\n✓ Created {len(unique_meds)} individual PNG plots in {output_dir}")


def create_hourly_aggregated_plot(df: pd.DataFrame,
                                   output_path: str = "output/final/sedation_hourly_dose.png") -> None:

    print("\nCreating hourly aggregated visualization...")
    
    df = df.copy()
    
    # Create hourly bins
    df['hour'] = df['admin_dttm'].dt.floor('H')
    
    # Determine grouping column
    group_col = 'med_category' if 'med_category' in df.columns else 'med_name'
    
    # Aggregate by hour and medication
    hourly_df = df.groupby(['hour', group_col, 'hospitalization_id'], as_index=False).agg({
        'med_dose': ['sum', 'mean', 'count']
    }).reset_index(drop=True)
    
    # Flatten column names
    hourly_df.columns = ['hour', group_col, 'hospitalization_id', 'total_dose', 'avg_dose', 'admin_count']
    
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create plot
    plot = (
        ggplot(hourly_df, aes(x='hour', y='total_dose', color='hospitalization_id'))
        + geom_line(alpha=0.6, size=0.5)
        + geom_point(alpha=0.4, size=1)
        + facet_wrap(f'~ {group_col}', scales='free_y', ncol=2)
        + labs(
            title='Hourly Total Sedation Dose by Medication',
            x='Hour',
            y='Total Dose per Hour',
            color='Hospitalization ID'
        )
        + theme_minimal()
        + theme(
            axis_text_x=element_text(angle=45, hjust=1, size=8),
            axis_text_y=element_text(size=8),
            strip_text=element_text(size=10),
            plot_title=element_text(size=14, face='bold'),
            legend_position='none'
        )
        + scale_x_datetime(date_labels='%Y-%m-%d %H:%M')
    )
    
    print(f"Saving plot to: {output_path}")
    ggsave(plot, output_path, width=14, height=10, dpi=300)
    print("Plot saved successfully!")


def create_duration_summary(periods_df: pd.DataFrame,
                            output_path: str = "output/final/sedation_duration_summary.png") -> None:
    """
    Create a summary visualization showing infusion durations.
    
    Parameters:
        periods_df (pd.DataFrame): DataFrame with infusion periods
        output_path (str): Path to save the plot
    """
    print("\nCreating duration summary visualization...")
    
    if 'duration_hours' not in periods_df.columns:
        print("Warning: duration_hours not found. Skipping duration summary.")
        return
    
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine grouping column
    group_col = 'med_category' if 'med_category' in periods_df.columns else 'med_name'
    
    # Create summary statistics
    summary_df = periods_df.groupby([group_col, 'hospitalization_id'], as_index=False).agg({
        'duration_hours': 'sum',
        'med_dose': 'mean'
    })
    
    plot = (
        ggplot(summary_df, aes(x=group_col, y='duration_hours', fill=group_col))
        + geom_boxplot(alpha=0.7)
        + geom_jitter(alpha=0.3, width=0.2)
        + labs(
            title='Total Infusion Duration by Medication',
            subtitle='Sum of all infusion periods per hospitalization',
            x='Medication',
            y='Total Duration (hours)',
            fill='Medication'
        )
        + theme_minimal()
        + theme(
            axis_text_x=element_text(angle=45, hjust=1, size=9),
            axis_text_y=element_text(size=9),
            plot_title=element_text(size=14, face='bold'),
            plot_subtitle=element_text(size=10),
            legend_position='none'
        )
    )
    
    print(f"Saving plot to: {output_path}")
    ggsave(plot, output_path, width=12, height=8, dpi=300)
    print("Plot saved successfully!")


def main():
    """
    Main function to run the visualization pipeline.
    Creates dose over time line graphs for sedation medications, grouped by hospitalization_id.
    """
    
    # File paths
    meds_parquet_file = "code/Release 1.0.0 (MIMIC-IV 3.1 - CLIF 2.1.0)/clif_medication_admin_continuous.parquet"
    vitals_parquet_file = "code/Release 1.0.0 (MIMIC-IV 3.1 - CLIF 2.1.0)/clif_vitals.parquet"
    cohort_csv_file = "code/cohort_hospitalization_MIMIC-IV.csv"
    
    # Optional: Limit number of hospitalizations for visualization (to avoid overcrowding)
    max_hospitalizations = None  # Set to None to show all, or specify a number (e.g., 50)
    
    try:
        # Load cohort hospitalization IDs from CSV file
        cohort_hosp_ids = load_cohort_hospitalization_ids(cohort_csv_file)
        
        # Read the medication parquet file
        df = read_parquet_file(meds_parquet_file)
        
        # Filter for sedation medications (med_group == 'sedation')
        sedation_df = filter_sedation_data(df)
        
        # Filter to only cohort hospitalizations
        initial_len = len(sedation_df)
        sedation_df = sedation_df[sedation_df['hospitalization_id'].isin(cohort_hosp_ids)].copy()
        print(f"\nFiltered to cohort hospitalizations: {len(sedation_df):,} rows (from {initial_len:,})")
        print(f"Cohort hospitalization IDs in sedation data: {sedation_df['hospitalization_id'].nunique():,} out of {len(cohort_hosp_ids):,} total cohort IDs")
        
        # Get list of hospitalization_ids for weight lookup (from cohort)
        hosp_ids_for_weights = sedation_df['hospitalization_id'].unique().tolist()
        
        # Load patient weights from vitals table
        weights_df = load_patient_weights(vitals_parquet_file, hospitalization_ids=hosp_ids_for_weights)
        
        # Convert weight-based doses (e.g., mcg/kg/min) to absolute doses (e.g., mcg/min)
        sedation_df = convert_weight_based_doses(sedation_df, weights_df)
        
        # Prepare data for visualization (handles start/stop/dose_change events)
        prepared_df = prepare_data_for_visualization(sedation_df)
        
        # Create visualizations
        
        # Create individual PNG plots for each sedative medication
        # Each plot shows individual patient trajectories (one line per hospitalization_id)
        create_individual_sedative_plots(
            prepared_df,
            output_dir="output/final/",
            sample_size=max_hospitalizations
        )
        
        print("\n" + "="*50)
        print("Visualization pipeline completed successfully!")
        print("="*50)
        print(f"\nTo visualize specific hospitalizations, set 'specific_hosp_ids' in main() function.")
        print(f"To show all hospitalizations, set 'max_hospitalizations = None' in main() function.")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

