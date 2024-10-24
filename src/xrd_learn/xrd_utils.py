"""
This module provides functions for loading, aligning, and processing X-ray diffraction (XRD) scan data. 
It supports loading scans from files, aligning peaks to specific values, and handling different types 
of input data for processing. This module is essential for preparing XRD data for visualization 
and further analysis.

Functions:
    - load_xrd_scan: Load a single XRD scan from a file.
    - load_xrd_scans: Load multiple XRD scans from a list of files.
    - align_peak_to_value: Align the peak of an XRD scan to a target value, with optional visualization.
    - process_input: Process input data of various types (file paths or pre-loaded data) to return 
      XRD scan arrays.

References:
    - https://matplotlib.org/stable/contents.html
    - https://xrayutilities.readthedocs.io/en/latest/
"""

import numpy as np
import matplotlib.pyplot as plt
import xrayutilities as xu
from typing import Union, List, Tuple
from scipy.signal import find_peaks

__author__ = "Joshua C. Agar, Yichen Guo"
__copyright__ = "Joshua C. Agar, Yichen Guo"
__license__ = "MIT"


def calculate_fwhm(X, Y, px):
    """
    Calculate the Full Width at Half Maximum (FWHM) for a given peak.

    Parameters:
    -----------
    X : array-like
        X-axis data (e.g., 2θ values).
    Y : array-like
        Y-axis data (e.g., intensity values).
    px : int
        x-value of the peak for which to calculate FWHM.

    Returns:
    --------
    fwhm : float
        Full Width at Half Maximum (in the same units as x).
    """

    # Peak height (maximum intensity) and half maximum
    peak_indices = int(np.where(X == px)[0])
    peak_height = Y[X==px]
    half_max = peak_height / 2

    # Find the left and right points where intensity crosses half maximum
    left_idx = np.where(Y[:peak_indices] <= half_max)[0][-1]
    right_idx = np.where(Y[peak_indices:] <= half_max)[0][0] + peak_indices

    # Calculate FWHM as the difference in x-values at half maximum
    x_fwhm = X[right_idx] - X[left_idx]
    y_fwhm = Y[right_idx]
    return x_fwhm, y_fwhm, X[left_idx], X[right_idx]  # Return FWHM and x-positions


def detect_peaks(x, y, num_peaks=3, prominence=0.1, distance=None):
    """
    Detects a specified number of peaks in the given x-y curve.

    Parameters:
    -----------
    x : array-like
        X-axis data (e.g., 2θ values).
    y : array-like
        Y-axis data (e.g., intensity values).
    num_peaks : int
        Number of top peaks to detect (default is 3).
    prominence : float
        Minimum prominence of peaks (default is 0.1).
    distance : float or None
        Minimum horizontal distance (in x-units) between peaks (optional).

    Returns:
    --------
    peak_x : list
        X-coordinates of detected peaks.
    peak_y : list
        Y-coordinates of detected peaks.
    """
    # Detect all peaks with the given prominence and distance
    peaks, properties = find_peaks(y, prominence=prominence, distance=distance)

    # Adjust number of peaks if fewer are detected
    num_detected = len(peaks)
    if num_detected < num_peaks:
        print(f"Warning: Only {num_detected} peaks detected, fewer than requested.")
        num_peaks = num_detected  # Use all available peaks


    # Sort peaks by prominence and select the top ones
    sorted_indices = sorted(range(len(peaks)), 
                            key=lambda i: properties['prominences'][i], 
                            reverse=True)[:num_peaks]
    sorted_peaks = [peaks[i] for i in sorted_indices]
    peak_x = [x[i] for i in sorted_peaks]
    peak_y = [y[i] for i in sorted_peaks]
    return peak_x, peak_y


def load_xrd_scan(file):
    """Load a single XRD scan from a file.

    This function reads an XRDML file and returns the scan data, including the 2-theta angles
    and corresponding intensities.

    Args:
        file (str): Path to the XRDML file to be loaded.
    
    Returns:
        tuple: A tuple containing X (2-theta angles) and Y (intensity values) arrays.
    """
    return xu.io.getxrdml_scan(file)


def load_xrd_scans(files):
    """Load multiple XRD scans from a list of files.

    This function loads XRD scans from multiple XRDML files, returning the angles (Xs), intensities (Ys), 
    and the length of each dataset for later processing or plotting.

    Args:
        files (list of str): A list of file paths for the XRDML files to be loaded.

    Returns:
        tuple: A tuple containing three lists:
            - Xs: List of 2-theta angles for each file.
            - Ys: List of intensity values for each file.
            - length_list: List of lengths of the 2-theta arrays for each file.
    """
    Xs, Ys, length_list = [], [], []
    for file in files:
        out = xu.io.getxrdml_scan(file)
        Xs.append(out[0])
        Ys.append(out[1])
        length_list.append(len(out[0]))
    return Xs, Ys, length_list


def align_peak_to_value(Xs, Ys, target_x_peak, viz=False):
    """Align the peak of XRD scans to a target value.

    This function shifts the 2-theta values of XRD scans so that the peak of each scan aligns with a target
    2-theta value. Optionally, a plot can be generated to visualize the original and shifted scans.

    Args:
        Xs (list of np.ndarray): List of 2-theta arrays for each scan.
        Ys (list of np.ndarray): List of intensity arrays for each scan.
        target_x_peak (float): The target 2-theta value to align the peaks to.
        viz (bool, optional): If True, a plot will be shown comparing the original and shifted scans. 
                              Default is False.
    
    Returns:
        tuple: The shifted Xs and the original Ys arrays.
    """
    for i, (X, Y) in enumerate(zip(Xs, Ys)):
        max_idx = np.argmax(Y)
        current_x_peak = X[max_idx]
        shift = target_x_peak - current_x_peak
        X_shifted = X + shift
        if viz:
            plt.figure(figsize=(8, 1))
            plt.plot(X, Y, color='tab:blue')
            plt.plot(X_shifted, Y, color='tab:orange')
            plt.axvline(current_x_peak, color='tab:blue', linestyle='--', linewidth=1, label=f'Original: {current_x_peak:.2f}')
            plt.axvline(target_x_peak, color='tab:orange', linestyle='--', linewidth=1, label=f'Target: {target_x_peak:.2f}')
            plt.yscale('log')
            plt.legend()
            plt.title(f'Peak: {current_x_peak:.4f} -> {target_x_peak:.4f}')
            plt.show()
        Xs[i] = X_shifted
    return Xs, Ys


def process_input(input_data: Union[str, List[str], Tuple[List, List, List]]):
    """
    Process input data for XRD analysis based on its type.

    This function handles three types of input:
    1. A single file path (string)
    2. A list of file paths
    3. A tuple containing Xs, Ys, and length_list (pre-loaded data)

    Args:
        input_data (Union[str, List[str], Tuple[List, List, List]]): 
            The input data to be processed. Can be:
            - A string representing a single file path.
            - A list of strings, each representing a file path.
            - A tuple of three lists (Xs, Ys, length_list).

    Returns:
        tuple: A tuple containing Xs, Ys, and length_list, either loaded from files or directly from the input.

    Raises:
        ValueError: If the input type is not recognized or if the tuple doesn't contain exactly three lists.
    
    Notes:
        - For file inputs, the function loads the data using the `load_xrd_scans` function.
        - If a single file path is provided, it's converted to a list before processing.
    """
    if isinstance(input_data, str):
        # Single file path
        return load_xrd_scans([input_data])
    elif isinstance(input_data, list):
        # List of file paths
        return load_xrd_scans(input_data)
    elif isinstance(input_data, tuple) and len(input_data) == 3:
        # Tuple of Xs, Ys, and length_list
        if all(isinstance(item, list) for item in input_data):
            return input_data
        else:
            raise ValueError("Tuple must contain three lists")
    else:
        raise ValueError("Invalid input type")