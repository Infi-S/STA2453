import pandas as pd
import numpy as np
from datetime import datetime
from load_dataset import load_from_path

def average_time_str(t1, t2):
    """
    Averages two time strings in the format '%H:%M:%S.%f' by computing the midpoint.
    """
    dt1 = datetime.strptime(t1.strip(), "%H:%M:%S.%f")
    dt2 = datetime.strptime(t2.strip(), "%H:%M:%S.%f")
    
    # Convert times to total seconds since midnight
    seconds1 = dt1.hour * 3600 + dt1.minute * 60 + dt1.second + dt1.microsecond / 1e6
    seconds2 = dt2.hour * 3600 + dt2.minute * 60 + dt2.second + dt2.microsecond / 1e6
    
    # Compute the average seconds
    avg_seconds = (seconds1 + seconds2) / 2.0
    
    # Convert average seconds back into hours, minutes, seconds, microseconds
    hours = int(avg_seconds // 3600)
    remainder = avg_seconds - hours * 3600
    minutes = int(remainder // 60)
    seconds = remainder - minutes * 60
    sec_int = int(seconds)
    microsec = int(round((seconds - sec_int) * 1e6))
    
    new_dt = datetime(1900, 1, 1, hour=hours, minute=minutes, second=sec_int, microsecond=microsec)
    return new_dt.strftime("%H:%M:%S.%f")

def generate_synthetic_sample_for_group(group, numeric_cols, noise_std = 1, init_stat = 0):
    """
    Generates one synthetic sample from the provided group:
      - Randomly selects two rows from the group.
      - Averages numeric columns and adds Gaussian noise.
      - Averages the 'Ping_time' column (computes midpoint).
      - Retains other non-numeric columns from the first sampled row.
    Returns the synthetic sample as a dictionary.
    """
    # Randomly select 2 rows (the global seed ensures reproducibility)
    sampled = group.sample(n=2, random_state = init_stat)
    
    # Average numeric columns and add Gaussian noise
    avg_numeric = sampled[numeric_cols].mean()
    noise = np.random.normal(loc=0.0, scale=noise_std, size=avg_numeric.shape)
    avg_numeric_noisy = avg_numeric + noise
    
    synthetic_sample = {}
    for col in group.columns:
        if col in numeric_cols:
            synthetic_sample[col] = avg_numeric_noisy[col]
        elif col == 'Ping_time':
            t1 = sampled.iloc[0]['Ping_time']
            t2 = sampled.iloc[1]['Ping_time']
            synthetic_sample[col] = average_time_str(t1, t2)
        else:
            # Retain non-numeric columns from the first sampled row
            synthetic_sample[col] = sampled.iloc[0][col]
            
    # synthetic_sample['isSynthetic'] = True
    return synthetic_sample

def generate_synthetic_samples(df, target_class, num_samples, noise_std = 1, init_stat = 0):
    """
    Generates a specified number of synthetic samples for the target class.
      - Filters the dataframe for the target class.
      - Groups data by 'fishNum' (only groups with at least 2 records are used).
      - Randomly picks a group (with replacement) and generates a synthetic sample.
    Returns a DataFrame with the synthetic samples.
    """
    df_class = df[df['Spe'] == target_class]
    # Keep groups that have at least 2 records
    groups = [group for _, group in df_class.groupby('fishNum') if len(group) >= 2]
    
    if not groups:
        print(f"No groups with at least 2 records found for class {target_class}.")
        return pd.DataFrame()
    
    # Identify numeric columns (used for averaging)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    synthetic_samples = []
    for _ in range(num_samples):
        # Randomly select one group (with replacement)
        group = groups[np.random.randint(len(groups))]
        synthetic_sample = generate_synthetic_sample_for_group(group, numeric_cols, noise_std = noise_std, init_stat = init_stat)
        init_stat = init_stat + 1
        synthetic_samples.append(synthetic_sample)
    
    return pd.DataFrame(synthetic_samples)

def generate_synthetic_dataset_from_path(data_path, samples_size = 20000, noise_std = 1, init_stat = 0):
    """
    Generates a complete synthetic dataset from a file path.
      - Load dataset from given path
      - For each speices generate synthetic samples of chosen samples size.
      - Append samples of each speices to a DataFrame
    Returns the DataFrame containing all the synthetic samples.
    """
    df = load_from_path(data_path)

    samples_per_class = {
        'LT': samples_size,   # synthetic samples for class 'LT'
        'SMB': samples_size   # synthetic samples for class 'SMB'
    }

    synthetic_samples_all = []
    for target_class, num_samples in samples_per_class.items():
        synthetic_df = generate_synthetic_samples(df, target_class, num_samples, noise_std = noise_std, init_stat = init_stat)
        synthetic_samples_all.append(synthetic_df)

    # Combine synthetic samples from all classes
    all_synthetic_df = pd.concat(synthetic_samples_all, ignore_index=True)

    # Combine synthetic samples with the original data to create an augmented dataset
    augmented_df = pd.concat([df, all_synthetic_df], ignore_index=True)

    return augmented_df

def generate_synthetic_dataset_from_dataFrame(df, samples_size = 20000, noise_std = 1, init_stat = 0):
    """
    Generates a complete synthetic dataset from a loaded dataframe.
      - For each speices generate synthetic samples of chosen samples size.
      - Append samples of each speices to a DataFrame
    Returns the DataFrame containing all the synthetic samples.
    """
    samples_per_class = {
        'LT': samples_size,   # synthetic samples for class 'LT'
        'SMB': samples_size   # synthetic samples for class 'SMB'
    }

    synthetic_samples_all = []
    for target_class, num_samples in samples_per_class.items():
        synthetic_df = generate_synthetic_samples(df, target_class, num_samples, noise_std = noise_std, init_stat = init_stat)
        synthetic_samples_all.append(synthetic_df)

    # Combine synthetic samples from all classes
    all_synthetic_df = pd.concat(synthetic_samples_all, ignore_index=True)

    # Combine synthetic samples with the original data to create an augmented dataset
    augmented_df = pd.concat([df, all_synthetic_df], ignore_index=True)

    return augmented_df
