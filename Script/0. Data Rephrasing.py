import pandas as pd

# Load and filter data (from previous steps)
df = pd.read_csv('ProcessedData/processed_AllFishCombined_unfiltered.csv')
specific_columns = {'fishNum', 'totalLength', 'weight', 'sex', 'airbladderTotalLength', 'Ping_time'}
columns_to_keep = []

for col in df.columns:
    if col in specific_columns:
        columns_to_keep.append(col)
    elif col.startswith('F'):
        try:
            f_value = float(col[1:])
            if 90 <= f_value and f_value <= 170:
                # print(f"Dropping empty column: {col}")
                continue
            else:
                columns_to_keep.append(col)
        except ValueError:
            pass

filtered_df = df[columns_to_keep].copy()

# Create columns that split the fishNum into speices code and index in species
filtered_df[['Spe', 'Index']] = filtered_df['fishNum'].str.extract(r'^(BUR|LT|LWF|SMB)(\d+)$')

# Convert the extracted numeric part to integer.
filtered_df['Index'] = filtered_df['Index'].astype(int)

# Save to new CSV
filtered_df.to_csv('ProcessedData/AllFishCombined_filtered.csv', index=False)

# validation
print("\nSample transformed fishNum entries:")
print(filtered_df.head())