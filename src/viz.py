import numpy as np
import matplotlib.pyplot as plt

def evaluate_image_histogram(image, outlier_std=3):
    """
    Generate a histogram of image pixel values with Z-score clipping and label mean, min, max, and std.
    
    Parameters:
    image (numpy array): The input image array. Assumes a grayscale image with values in range 0-255.
    z_thresh (float): The Z-score threshold for clipping.
    """
    # Flatten the image to a 1D array
    pixel_values = image.flatten()
    
    # Calculate mean and standard deviation
    mean_val = np.mean(pixel_values)
    std_val = np.std(pixel_values)
    
    # Clip values based on Z-score threshold
    lower_clip = mean_val - outlier_std * std_val
    upper_clip = mean_val + outlier_std * std_val
    clipped_values = pixel_values[(pixel_values >= lower_clip) & (pixel_values <= upper_clip)]
    
    # Calculate statistics on clipped values
    mean_clipped = np.mean(clipped_values)
    min_clipped = np.min(clipped_values)
    max_clipped = np.max(clipped_values)
    std_clipped = np.std(clipped_values)
    
    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(clipped_values, bins=100, range=(lower_clip, upper_clip), alpha=0.3, edgecolor='black')
    plt.title(f'Image Histogram (removing noise outside ±{outlier_std}σ)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    # Add text for the statistics
    plt.axvline(mean_clipped, color='red', linestyle='dashed', linewidth=2)
    plt.text(mean_clipped + np.abs(mean_clipped)*0.1, plt.ylim()[1] * 0.9, f'Mean: {mean_clipped:.2e}\nStd: {std_clipped:.2e}', color='black')
    
    plt.axvline(min_clipped, color='green', linestyle='dashed', linewidth=2)
    plt.text(min_clipped + np.abs(min_clipped)*0.05, plt.ylim()[1] * 0.8, f'Min:\n{min_clipped:.2e}', color='green')
    
    plt.axvline(max_clipped, color='green', linestyle='dashed', linewidth=2)
    plt.text(max_clipped + np.abs(max_clipped)*0.05, plt.ylim()[1] * 0.8, f'Max:\n{max_clipped:.2e}', color='green')
    
    for i in range(3):
        lb, hb = mean_clipped-std_clipped*(i+1), mean_clipped+std_clipped*(i+1)
        plt.axvline(lb, color='blue', linestyle='dashed', linewidth=2, alpha=0.5)
        plt.text(lb + np.abs(lb)*0.05, plt.ylim()[1] * (0.4+i/10), f'-{i+1}σ:\n{lb:.2e}', color='black', alpha=0.8)
        plt.axvline(hb, color='blue', linestyle='dashed', linewidth=2, alpha=0.5)
        plt.text(hb + np.abs(hb)*0.05, plt.ylim()[1] * (0.4+i/10), f'+{i+1}σ:\n{hb:.2e}', color='black', alpha=0.8)
    
    # Show plot
    plt.show()
