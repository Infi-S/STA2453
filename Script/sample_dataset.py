"""
This script creates a reproducible sample dataset from a large CSV by randomly selecting
a specified number of rows for each of two species categories ('LT' and 'SMB') in the
'Spe' column.

Steps:
1. Import pandas.
2. Set a random seed for reproducibility.
3. Specify how many rows to sample for each species.
4. Load the full dataset into a DataFrame.
5. Filter and sample for each species.
6. Combine the samples, save to CSV, and preview the result.
"""

# 1. Import relevant packages
import pandas as pd
import os

# 2. Set random seed for reproducible sampling
seed = 0

# 3. Define sample sizes for each species
number_per_class = 500    # number of rows to sample for each Species

# 4. Load the full dataset and drop the unwanted individuals
file_path = os.path.join('Data', 'AllFishCombined_filtered.csv')
df = pd.read_csv(file_path, low_memory=False)
df = df[~df['fishNum'].isin(['LT008', 'LT016'])]

# 5. Filter rows where Spe is 'LT' or 'SMB'
df_lt = df[df['Spe'] == 'LT']
df_smb = df[df['Spe'] == 'SMB']

# 6a. Sample the specified number of rows for each species
sample_lt = df_lt.sample(n=number_per_class, random_state=seed)
sample_smb = df_smb.sample(n=number_per_class, random_state=seed)

# 6b. Combine the two sampled subsets into one DataFrame
sample_df = pd.concat([sample_lt, sample_smb]).reset_index(drop=True)

# 7. Parse 'Ping_time' column as datetime for accurate sorting
sample_df['Ping_time'] = pd.to_datetime(
    sample_df['Ping_time'],
    format=' %H:%M:%S.%f',
    errors='coerce'  # ensures any malformed values become NaT
)

# 8. Sort by 'fishNum' first, then by 'Ping_time'
sample_df = sample_df.sort_values(
    by=['fishNum', 'Ping_time']
).reset_index(drop=True)

# 9. Convert 'Ping_time' back to string if you need the original format
sample_df['Ping_time'] = sample_df['Ping_time'].dt.strftime(' %H:%M:%S.%f')

# 10. Save the sampled dataset to a new CSV
output_path = os.path.join('Data', 'sample_data.csv')
sample_df.to_csv(output_path, index=False)