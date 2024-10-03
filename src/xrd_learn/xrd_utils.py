import numpy as np
import matplotlib.pyplot as plt
import xrayutilities as xu
from typing import Union, List, Tuple

def load_xrd_scan(file):
    return xu.io.getxrdml_scan(file)

def load_xrd_scans(files):
    Xs, Ys, length_list = [], [], []
    for file in files:
        out = xu.io.getxrdml_scan(file)
        Xs.append(out[0])
        Ys.append(out[1])
        length_list.append(len(out[0]))
    return Xs, Ys, length_list


def align_peak_to_value(Xs, Ys, target_x_peak, viz=False):
    for i, (X, Y) in enumerate(zip(Xs, Ys)):
        max_idx = np.argmax(Y)
        current_x_peak = X[max_idx]
        shift = target_x_peak - current_x_peak
        X_shifted = X + shift
        if viz:
            plt.figure(figsize=(8,1))
            plt.plot(X, Y)
            plt.plot(X_shifted, Y)
            plt.axvline(target_x_peak, color='orange', linestyle='--', linewidth=1, label=f'target: {target_x_peak:.2f}')
            plt.axvline(current_x_peak, color='blue', linestyle='--', linewidth=1, label=f'original: {current_x_peak:.2f}')
            plt.yscale('log')
            plt.legend()
            plt.title(f'peak: {current_x_peak:.2f} -> {target_x_peak:.2f}')
            plt.show()
        Xs[i] = X_shifted
    return Xs, Ys


def process_input(input_data: Union[str, List[str], Tuple[List, List, List]]):
    """
    Process input data for function A based on its type.

    This function handles three types of inputs:
    1. A single file path (string)
    2. A list of file paths
    3. A tuple containing Xs, Ys, and length_list

    Parameters:
    -----------
    input_data : Union[str, List[str], Tuple[List, List, List]]
        The input data to be processed. Can be one of the following:
        - A string representing a single file path
        - A list of strings, each representing a file path
        - A tuple of three lists (Xs, Ys, length_list)

    Returns:
    --------
    Tuple[List, List, List]
        A tuple containing Xs, Ys, and length_list, either loaded from files or directly from the input.

    Raises:
    -------
    ValueError
        If the input type is not recognized or if the tuple doesn't contain exactly three lists.

    Notes:
    ------
    - For file inputs, the function uses an external function B to load data from files.
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
