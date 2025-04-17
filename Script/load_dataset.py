import pandas as pd
import numpy as np

def load_from_path(data_path, target_classes = [], exclude_individuals = []):
    """
    Load a dataset from given file path and pre-process it:
      - Load file from given path.
      - Reorder the columns for a better overview.
      - Filter to keep only the target classes if specified.
      - Exclude fishNum that does not want.
    Returns a DataFrame in the expected format, containing only relevant information.
    """
    # Load the dataset
    df = pd.read_csv(data_path, low_memory=False)
    df = df.drop(columns=['airbladderTotalLength', 'totalLength', 'weight', 'sex'], errors='ignore')

    # Reorder the columns
    new_order = ['fishNum', 'Spe', 'Index'] + [col for col in df.columns if col not in ['fishNum', 'Spe', 'Index']]
    df = df[new_order]

    # Filter to the target classes
    if (target_classes):
        df = df[df['Spe'].isin(target_classes)]

    # Mannually exclude fishNum that does not want if avaiable
    if (exclude_individuals):
        df = df[~df['fishNum'].isin(exclude_individuals)]

    return df
