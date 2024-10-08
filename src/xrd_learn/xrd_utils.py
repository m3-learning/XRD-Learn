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

__author__ = "Joshua C. Agar, Yichen Guo"
__copyright__ = "Joshua C. Agar, Yichen Guo"
__license__ = "MIT"


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
            plt.plot(X, Y)
            plt.plot(X_shifted, Y)
            plt.axvline(target_x_peak, color='orange', linestyle='--', linewidth=1, label=f'Target: {target_x_peak:.2f}')
            plt.axvline(current_x_peak, color='blue', linestyle='--', linewidth=1, label=f'Original: {current_x_peak:.2f}')
            plt.yscale('log')
            plt.legend()
            plt.title(f'Peak: {current_x_peak:.2f} -> {target_x_peak:.2f}')
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