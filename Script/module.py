import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def interpolate_ping_time(time_str1: str, time_str2: str, lam: float) -> str:
    """
    Interpolate between two time strings using a linear factor lam.
    Both time_str1 and time_str2 should be in the format '%H:%M:%S.%f'.
    Returns the interpolated time as a string in the same format.
    """
    t1 = datetime.strptime(time_str1.strip(), '%H:%M:%S.%f')
    t2 = datetime.strptime(time_str2.strip(), '%H:%M:%S.%f')
    seconds1 = t1.hour * 3600 + t1.minute * 60 + t1.second + t1.microsecond / 1e6
    seconds2 = t2.hour * 3600 + t2.minute * 60 + t2.second + t2.microsecond / 1e6
    synthetic_seconds = seconds1 + lam * (seconds2 - seconds1)
    synthetic_time = datetime(1900, 1, 1) + timedelta(seconds=synthetic_seconds)
    return synthetic_time.strftime('%H:%M:%S.%f')

def generate_synthetic_sample(row, neighbor_row, lam: float) -> dict:
    """
    Generate one synthetic sample by linearly interpolating between two rows using factor lam.
    The 'Ping_time' column is interpolated using the interpolate_ping_time function.
    For categorical columns ('Spe', 'fishNum'), the original value is retained.
    """
    synthetic_sample = {}
    for col in row.index:
        if col == 'Ping_time':
            synthetic_sample['Ping_time'] = interpolate_ping_time(row['Ping_time'], neighbor_row['Ping_time'], lam)
        elif col in ['Spe', 'fishNum']:
            synthetic_sample[col] = row[col]
        else:
            try:
                val1 = float(row[col])
                val2 = float(neighbor_row[col])
                synthetic_sample[col] = val1 + lam * (val2 - val1)
            except (ValueError, TypeError):
                synthetic_sample[col] = row[col]
    return synthetic_sample

def generate_synthetic_sample_with_noise(row, neighbor_row, lam: float, noise_std: float = 0.01) -> dict:
    """
    Generate one synthetic sample by linearly interpolating between two rows using factor lam,
    then add normally distributed noise to numeric features.
    
    The 'Ping_time' column is interpolated using interpolate_ping_time.
    For categorical columns ('Spe', 'fishNum'), the original value is retained.
    The noise_std parameter controls the standard deviation of the added Gaussian noise.
    """
    synthetic_sample = {}
    for col in row.index:
        if col == 'Ping_time':
            synthetic_sample['Ping_time'] = interpolate_ping_time(row['Ping_time'], neighbor_row['Ping_time'], lam)
        elif col in ['Spe', 'fishNum']:
            synthetic_sample[col] = row[col]
        else:
            try:
                val1 = float(row[col])
                val2 = float(neighbor_row[col])
                interpolated_val = val1 + lam * (val2 - val1)
                # Add Gaussian noise
                noise = np.random.normal(0, noise_std)
                synthetic_sample[col] = interpolated_val + noise
            except (ValueError, TypeError):
                synthetic_sample[col] = row[col]
    return synthetic_sample

def generate_synthetic_samples_for_class(df, spe_value: str, desired_sample_size: int, noise_std: float = 1):
    """
    Generate synthetic samples for a specified class (Spe) using random interpolation with noise.
    
    Parameters:
      df: The input DataFrame that has been filtered for relevant classes.
      spe_value: The value of 'Spe' (e.g., 'LT' or 'SMB') for which to generate synthetic samples.
      desired_sample_size: The total number of synthetic samples to generate.
      noise_std: Standard deviation of the Gaussian noise added to numeric features.
    
    Returns:
      A DataFrame containing the synthetic samples.
    """
    # Filter the DataFrame for the specified class
    class_df = df[df['Spe'] == spe_value]
    synthetic_samples = []
    
    # Continue generating until we have the desired number of synthetic samples
    while len(synthetic_samples) < desired_sample_size:
        # Randomly select one fishNum from the filtered class data
        fish_candidates = class_df['fishNum'].unique()
        chosen_fish = np.random.choice(fish_candidates)
        
        # Get all rows corresponding to the chosen fishNum
        group = class_df[class_df['fishNum'] == chosen_fish].reset_index(drop=True)
        
        # Need at least two rows to perform interpolation
        if len(group) < 2:
            continue
        
        # Randomly select two distinct rows from the group
        indices = np.random.choice(len(group), size=2, replace=False)
        row1 = group.iloc[indices[0]]
        row2 = group.iloc[indices[1]]
        
        lam = np.random.rand()  # Random interpolation factor between 0 and 1
        synthetic_sample = generate_synthetic_sample_with_noise(row1, row2, lam, noise_std)
        synthetic_samples.append(synthetic_sample)
    
    return pd.DataFrame(synthetic_samples)