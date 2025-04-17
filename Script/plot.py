import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_feature_importance(df, title = 'Feature Importance Plot', cap = 3):
    """
    Plots feature importance values in a line plot from the given DataFrame.
    
    Parameters:
        df (pd.DataFrame): A DataFrame containing at least two columns:
            'Feature' : names of the features
            'Importance' : the calculated importance values for each feature.
            
    The x-axis will show the feature names and the y-axis will show their importance.
    The plot is displayed using matplotlib.
    """
    # Ensure the DataFrame is sorted by importance (optional)
    df = df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 3))
    plt.plot(df['Feature'], df['Importance'], marker='o', linestyle='-')
    # plt.xticks(rotation=90)
    plt.xticks([])       
    plt.ylim(0, cap)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_spectrogram(spec, title="Spectrogram", cmap='viridis'):
    """
    Plots a single spectrogram.
    
    Parameters:
        spec (np.ndarray or torch.Tensor): The spectrogram to be plotted. This can be:
            - A 2D array with shape (height, width), or 
            - A 3D array with one channel, e.g., (1, height, width) or (height, width, 1).
        title (str, optional): The title of the plot.
        cmap (str, optional): The matplotlib colormap to use (default is 'viridis').
    """
    # If the input is a torch.Tensor, convert it to a numpy array.
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().cpu().numpy()
    
    # If the input is 3D and one of the dimensions is 1, squeeze that dimension.
    if spec.ndim == 3:
        # Check if the first dimension is singleton.
        if spec.shape[0] == 1:
            spec = np.squeeze(spec, axis=0)
        # Or, if the last dimension is singleton.
        elif spec.shape[-1] == 1:
            spec = np.squeeze(spec, axis=-1)
        else:
            # Otherwise, default to using the first channel.
            spec = spec[0]
    
    plt.figure(figsize=(12, 4))
    plt.imshow(spec, aspect='auto', origin='lower', cmap=cmap)
    if title is not None:
        plt.title(title)
    plt.colorbar()
    plt.xlabel("Normalized Time Steps")
    plt.ylabel("Frequency Bins")
    plt.xticks([])  # Remove X-axis ticks for a neater look.
    plt.show()


def plot_learning_curve(ax, train_losses, val_losses, fold_number):
    """
    Plots the learning curve (train and validation losses) for a given fold
    on the provided axes instance 'ax' and removes X-axis ticks.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to plot on.
        train_losses (list or array): The training losses per epoch.
        val_losses (list or array): The validation losses per epoch.
        fold_number (int): The fold number (for titling the plot).
    """
    epochs = len(train_losses)
    x = range(1, epochs + 1)
    ax.plot(x, train_losses, label="Train Loss")
    ax.plot(x, val_losses, label="Validation Loss")
    ax.set_title(f'Fold {fold_number}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xticks([])
    ax.legend()

def plot_fold_accuracies(fold_accuracies, title = 'Final Accuracies for Each Fold'):
    """
    Plots a bar chart of final accuracies for each fold.

    Parameters:
        fold_accuracies (list or array): A list/array of final accuracies for each fold.
    """
    fold_numbers = np.arange(1, len(fold_accuracies) + 1)
    avg_accuracy = np.mean(fold_accuracies)

    plt.figure(figsize=(10, 4))
    bars = plt.bar(fold_numbers, fold_accuracies, color='skyblue', edgecolor='k')

    # Optional: add text labels above each bar with the accuracy values.
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2f}', 
                 ha='center', va='bottom', fontsize=10)

    # Plot an average accuracy horizontal line.
    plt.axhline(y=avg_accuracy, color='red', linestyle='--', label=f'Average Accuracy = {avg_accuracy:.2f}')
    plt.xlabel("Fold Number")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xticks(fold_numbers)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()