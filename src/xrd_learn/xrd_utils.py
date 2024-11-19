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
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

__author__ = "Joshua C. Agar, Yichen Guo"
__copyright__ = "Joshua C. Agar, Yichen Guo"
__license__ = "MIT"


# def calculate_fwhm(X, Y, px, viz=False):
#     """
#     Calculate the Full Width at Half Maximum (FWHM) for a given peak.

#     Parameters:
#     -----------
#     X : array-like
#         X-axis data (e.g., 2θ values).
#     Y : array-like
#         Y-axis data (e.g., intensity values).
#     px : int
#         x-value of the peak for which to calculate FWHM.

#     Returns:
#     --------
#     fwhm : float
#         Full Width at Half Maximum (in the same units as x).
#     """

#     # Peak height (maximum intensity) and half maximum
#     peak_indices = int(np.where(X == px)[0])
#     peak_height = Y[X==px]
#     half_max = peak_height / 2

#     # Find the left and right points where intensity crosses half maximum
#     left_idx = np.where(Y[:peak_indices] <= half_max)[0][-1]
#     right_idx = np.where(Y[peak_indices:] <= half_max)[0][0] + peak_indices

#     # Calculate FWHM as the difference in x-values at half maximum
#     x_fwhm = X[right_idx] - X[left_idx]
#     y_fwhm = Y[right_idx]
    
#     if viz:
#         plt.figure(figsize=(8, 2))
#         plt.plot(X, Y, label='Data')
#         plt.axhline(half_max, color='gray', linestyle='--', label=f'Half Max = {half_max[0]:.4f}')
#         plt.axvline(X[left_idx], color='orange', linestyle='--', label=f'Left FWHM = {X[left_idx]:.4f}')
#         plt.axvline(X[right_idx], color='purple', linestyle='--', label=f'Right FWHM = {X[right_idx]:.4f}')
#         plt.axvline(px, color='red', linestyle='-', label=f'Peak = {px}')
#         plt.scatter([X[left_idx], X[right_idx]], [Y[left_idx], Y[right_idx]], color='black', zorder=5)
#         plt.xlabel('X')
#         plt.ylabel('Y')
#         plt.title(f'FWHM Calculation (FWHM = {x_fwhm:.4f})')
#         plt.legend()
#         plt.show()
        
#     return x_fwhm, y_fwhm, X[left_idx], X[right_idx]  # Return FWHM and x-positions


def upsample_XY(X, Y, num_points=5000):
    interp_func = interp1d(X, Y, kind='cubic')
    # Define a finer grid for upsampling
    X = np.linspace(np.min(X), np.max(X), num_points)  # New x with 100 points
    Y = interp_func(X)  # Interpolated y values on the finer grid
    return X, Y
    
    
# Define Gaussian and Lorentzian functions for fitting
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def lorentzian(x, a, x0, gamma):
    return a * gamma**2 / ((x - x0)**2 + gamma**2)

def calculate_fwhm(X, Y, px, fit_type='gaussian', viz=False):
    """
    Calculate the Full Width at Half Maximum (FWHM) for a given peak using Gaussian or Lorentzian fit.

    Parameters:
    -----------
    X : array-like
        X-axis data (e.g., 2θ values).
    Y : array-like
        Y-axis data (e.g., intensity values).
    px : int
        x-value of the peak for which to calculate FWHM.
    fit_type : str, optional
        Type of fit to use ('gaussian' or 'lorentzian'). Default is 'gaussian'.
    viz : bool, optional
        If True, a plot will be shown with the fit and FWHM. Default is False.

    Returns:
    --------
    fwhm : float
        Full Width at Half Maximum (in the same units as x).
    """
    # Select the fitting function based on fit_type
    if fit_type == 'gaussian':
        fit_func = gaussian
    elif fit_type == 'lorentzian':
        fit_func = lorentzian
    else:
        raise ValueError("fit_type must be 'gaussian' or 'lorentzian'")

    # Initial guesses for the parameters
    a_guess = np.max(Y)
    x0_guess = X[np.argmax(Y)]
    sigma_guess = (X[-1] - X[0]) / 4

    # Fit the data
    try:
        params, _ = curve_fit(fit_func, X, Y, p0=[a_guess, x0_guess, sigma_guess])
    except RuntimeError:
        print("Fit did not converge.")
        return None, None, None, None

    # Extract the fitted parameters
    a, x0, width_param = params

    # Calculate FWHM based on the fitting function
    if fit_type == 'gaussian':
        fwhm = 2 * np.sqrt(2 * np.log(2)) * width_param  # FWHM for Gaussian
    elif fit_type == 'lorentzian':
        fwhm = 2 * width_param  # FWHM for Lorentzian
    fwhm = abs(fwhm)  # Ensure FWHM is positive

    if viz:
        plt.figure(figsize=(8, 4))
        plt.plot(X, Y, 'o', label='Data')
        plt.plot(X, fit_func(X, *params), '-', label=f'{fit_type.capitalize()} Fit')
        plt.axhline(a / 2, color='gray', linestyle='--', label=f'Half Max = {a / 2:.4f}')
        plt.axvline(x0 - fwhm / 2, color='orange', linestyle='--', label=f'Left FWHM = {x0 - fwhm / 2:.4f}')
        plt.axvline(x0 + fwhm / 2, color='purple', linestyle='--', label=f'Right FWHM = {x0 + fwhm / 2:.4f}')
        plt.scatter([x0 - fwhm / 2, x0 + fwhm / 2], [a / 2, a / 2], color='black', zorder=5)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'FWHM Calculation (FWHM = {fwhm:.4f})')
        plt.legend()
        plt.show()

    return fwhm, a, x0 - fwhm / 2, x0 + fwhm / 2  # Return FWHM and x-positions of FWHM


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
            plt.axvline(current_x_peak, color='tab:blue', linestyle='--', linewidth=1, label=f'Original: {current_x_peak:.4f}')
            plt.axvline(target_x_peak, color='tab:orange', linestyle='--', linewidth=1, label=f'Target: {target_x_peak:.4f}')
            plt.yscale('log')
            plt.legend()
            plt.title(f'Peak: {current_x_peak:.4f} -> {target_x_peak:.4f}')
            plt.show()
        Xs[i] = X_shifted
    return Xs, Ys

def align_fwhm_center_to_value(Xs, Ys, target_x_peak, viz=False):
    """
    Align the center of the Full Width at Half Maximum (FWHM) of XRD scans to a target value.
    
    This function shifts the 2-theta values of XRD scans so that the center of the FWHM of each scan
    aligns with a target 2-theta value. Optionally, a plot can be generated to visualize the original
    and shifted scans.
    
    Args:
        Xs (list of np.ndarray): List of 2-theta arrays for each scan.
        Ys (list of np.ndarray): List of intensity arrays for each scan.
        target_x_peak (float): The target 2-theta value to align the FWHM centers to.
        viz (bool, optional): If True, a plot will be shown comparing the original and shifted scans.
                              Default is False.
    Returns:
        tuple: The shifted Xs and the original Ys arrays.
    """
    fwhm_list = []
    for i, (X, Y) in enumerate(zip(Xs, Ys)):
        max_idx = np.argmax(Y)
        current_x_peak = X[max_idx]
        fwhm, y_fwhm, x_left, x_right = calculate_fwhm(X, Y, current_x_peak)
        current_x_peak = (x_left + x_right) / 2
        shift = target_x_peak - current_x_peak
        X_shifted = X + shift
        Xs[i] = X_shifted
        fwhm_list.append(fwhm)
        if viz:
            plt.figure(figsize=(8, 1))
            plt.plot(X, Y, color='tab:blue')
            plt.plot(X_shifted, Y, color='tab:orange')
            plt.axvline(current_x_peak, color='tab:blue', linestyle='--', linewidth=1, label=f'Original: {current_x_peak:.4f}')
            plt.axvline(target_x_peak, color='tab:orange', linestyle='--', linewidth=1, label=f'Target: {target_x_peak:.4f}')
            plt.yscale('log')
            plt.legend()
            plt.title(f'Peak: {current_x_peak:.4f} -> {target_x_peak:.4f}')
            plt.show()
    return Xs, Ys, fwhm_list

def align_peak_y_to_value(Xs, Ys, target_y_peak=None, use_global_max=False, viz=False):
    """Align the peak intensity (Y value) of XRD scans to a target value.

    This function scales the intensity values of each scan so that the maximum intensity 
    aligns with a target Y peak value or the global maximum Y value across all scans. 
    Optionally, a plot can be generated to visualize the original and scaled scans.

    Args:
        Xs (list of np.ndarray): List of 2-theta arrays for each scan.
        Ys (list of np.ndarray): List of intensity arrays for each scan.
        target_y_peak (float, optional): The target intensity value to align the peaks to.
                                         Ignored if `use_global_max` is True.
        use_global_max (bool, optional): If True, uses the maximum Y value across all scans as the target.
                                         Default is False.
        viz (bool, optional): If True, a plot will be shown comparing the original and scaled scans. 
                              Default is False.
    
    Returns:
        tuple: The original Xs arrays and the scaled Ys arrays.
    """
    # Determine the target Y peak value
    if use_global_max:
        target_y_peak = max(np.max(Y) for Y in Ys)
    
    if target_y_peak is None:
        raise ValueError("Either target_y_peak must be specified, or use_global_max must be True.")
    
    # Scale each Y array to align its peak with the target Y peak value
    for i, (X, Y) in enumerate(zip(Xs, Ys)):
        max_y = np.max(Y)
        scale_factor = target_y_peak / max_y
        Y_scaled = Y * scale_factor
        if viz:
            plt.figure(figsize=(8, 4))
            plt.plot(X, Y, color='tab:blue', label='Original')
            plt.plot(X, Y_scaled, color='tab:orange', label='Scaled')
            plt.axhline(max_y, color='tab:blue', linestyle='--', linewidth=1, label=f'Original Peak Y: {max_y:.4f}')
            plt.axhline(target_y_peak, color='tab:orange', linestyle='--', linewidth=1, label=f'Target Y Peak: {target_y_peak:.4f}')
            plt.yscale('log')
            plt.legend()
            plt.title(f'Y Peak Aligned: {max_y:.4f} -> {target_y_peak:.4f}')
            plt.show()
        Ys[i] = Y_scaled
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