import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_single_spectrogram(df, spe, desired_length = 50,init_stat = 0):
    """
    Generates ONE spectrogram by:
      1) Randomly picking one fishNum from the given species (spe).
      2) Randomly sampling 'desired_length' rows from that fish.
      3) Converting Ping_time to datetime, sorting, and normalizing time.
      4) Interpolating the frequency columns onto a [0..1] grid of size 'desired_length'.
    
    Returns:
      spectrogram (np.ndarray of shape (desired_length, num_freqs))
    """
    # Identify frequency columns
    freq_cols = [col for col in df.columns if col.startswith('F')]
    
    # Subset to only rows of this species
    df_spe = df[df['Spe'] == spe]
    
    # Get unique fish within this species
    fish_list = df_spe['fishNum'].unique()
    
    # Randomly pick one fish
    chosen_fish = np.random.choice(fish_list)
    
    # Filter dataframe to the chosen fish
    df_fish = df_spe[df_spe['fishNum'] == chosen_fish].copy()
    
    # Randomly sample 'desired_length' rows (with replacement if not enough rows)
    if len(df_fish) < desired_length:
        df_sampled = df_fish.sample(n=desired_length, replace=True, random_state=init_stat)
    else:
        df_sampled = df_fish.sample(n=desired_length, replace=False, random_state=init_stat)
    
    # Convert ping times to datetime
    df_sampled['Ping_time_dt'] = pd.to_datetime(df_sampled['Ping_time'], format= " %H:%M:%S.%f")
    # Sort by chronological order
    df_sampled = df_sampled.sort_values('Ping_time_dt')
    
    # Convert to "seconds since first ping"
    times = df_sampled['Ping_time_dt']
    t_min = times.iloc[0]
    times_sec = times.apply(lambda x: (x - t_min).total_seconds()).values
    
    # Normalize time from 0 to 1
    t_min_val = times_sec.min()
    t_max_val = times_sec.max()
    if t_max_val == t_min_val:
        # Edge case: all times identical
        norm_times = np.zeros_like(times_sec)
    else:
        norm_times = (times_sec - t_min_val) / (t_max_val - t_min_val)
    
    # Extract frequency data
    freq_data = df_sampled[freq_cols].values  # shape: (desired_length, num_freqs)
    num_freqs = freq_data.shape[1]
    
    # Prepare interpolation grid
    desired_grid = np.linspace(0, 1, num=desired_length)
    
    # Interpolate each frequency column
    spectrogram = np.zeros((desired_length, num_freqs))
    
    # Another edge case: if norm_times are all the same value
    if np.all(norm_times == norm_times[0]):
        # Just repeat the first row of freq_data
        spectrogram = np.repeat(freq_data[0:1, :], desired_length, axis=0)
    else:
        for j in range(num_freqs):
            spectrogram[:, j] = np.interp(desired_grid, norm_times, freq_data[:, j])
    
    return spectrogram


def get_class_spectrograms(df, spe, desired_length = 50, iteration_for_class = 10, init_stat = 0):
    """
    Generates multiple spectrograms for a single species (spe).
    Iterates 'iteration_for_class' times, each time calling get_single_spectrogram.
    
    Returns:
      spectrogram_list (list of np.ndarrays), each of shape (desired_length, num_freqs)
    """
    spectrogram_list = []
    for _ in range(iteration_for_class):
        spec = get_single_spectrogram(df, spe, desired_length, init_stat)
        init_stat += 1
        spectrogram_list.append(spec)
    
    return spectrogram_list        

def save_spectrograms(spectrogram_array, prefix = "Spe_", num_to_save = 1, folder = "spectrograms"):
    """
    Save a specified number of spectrograms from an array to a local folder.
    
    Parameters:
        spectrogram_array (list or np.ndarray): 
            An array or list of 2D spectrograms (each should be a 2D numpy array).
        prefix (str): 
            The prefix to use for the saved file names.
        num_to_save (int): 
            The number of spectrogram images to save.
        folder (str): 
            The target folder to save images. Default is "spectrograms".
    """
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Ensure we do not exceed the available number of spectrograms
    num_to_save = min(num_to_save, len(spectrogram_array))
    
    for i in range(num_to_save):
        # Get the spectrogram (assumed to be a 2D numpy array)
        spec = spectrogram_array[i]
        # Create a filename using the prefix and index
        filename = os.path.join(folder, f"{prefix}{i}.png")
        # Save the spectrogram image; change cmap if desired (e.g., 'viridis', 'gray')
        # plt.imsave(filename, spec, cmap='viridis')

        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot the spectrogram ensuring a square aspect ratio.
        # Using 'aspect' parameter ensures the image area is square.
        cax = ax.imshow(spec, cmap='viridis', aspect='equal')
        
        # Set the x and y labels.
        ax.set_xlabel("Time (normalized)")
        ax.set_ylabel("Frequency")
        
        # Add a colorbar with the label "Amplitude"
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label("Amplitude")
        
        # Force the axis area to be square (optional if aspect='equal' already does it)
        ax.set_aspect('equal')
        
        # Adjust layout and save the figure
        plt.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
        print(f"Saved {filename}")
