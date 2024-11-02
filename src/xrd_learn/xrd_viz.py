"""
This module provides functions for visualizing X-ray diffraction (XRD) data, including plotting XRD scans. These visualizations are useful for analyzing the structural properties of 
materials through XRD experiments.

Functions:
    - plot_xrd: Plots XRD scans for different datasets, allowing for customizations such as padding sequences, 
      logarithmic scales, and figure saving options.

References:
    - https://matplotlib.org/stable/tutorials/index.html
    - https://xrayutilities.readthedocs.io/en/latest/
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from xrd_learn.xrd_utils import process_input

__author__ = "Joshua C. Agar, Yichen Guo"
__copyright__ = "Joshua C. Agar, Yichen Guo"
__license__ = "MIT"


def plot_xrd(inputs, labels, title='XRD Scan', xrange=None, yrange=None, diff=1e3, fig=None, ax=None, figsize=(6,4), 
             legend_style='legend', colors=colormaps.get_cmap('tab10'), text_offset_ratio=None, grid=False, pad_sequence=[], filename=None):
    
    """Plot XRD scans for multiple datasets.

    This function generates a plot for XRD scans, supporting multiple datasets and applying
    customizable options for padding, logarithmic scale, and saving the figure to file.

    Args:
        inputs (list): A list of input datasets to be processed and plotted.
        labels (list): A list of labels corresponding to the datasets.
        title (str, optional): Title of the plot. Default is 'XRD Scan'.
        xrange (tuple, optional): The x-axis range for the plot.
        diff (float, optional): Scaling factor between different datasets. Default is 1e3.
        fig (matplotlib.figure.Figure, optional): Custom figure for the plot.
        ax (matplotlib.axes.Axes, optional): Custom axes for the plot.
        figsize (tuple, optional): Figure size. Default is (6, 4).
        legend_style (str, optional): Style of the legend. Default is 'legend', options: 'legend', 'label.
        colors (list, optional): List of colors for the datasets. Default is None.
        text_offset_ratio (tuple, optional): Offset ratio for text labels. Default is None. (x_offset, y_offset).
        grid (bool, optional): Whether to show gridlines. Default is False.
        pad_sequence (list, optional): Sequence for padding datasets if they have different ranges.
        filename (str, optional): If provided, the plot will be saved with the given filename.
    
    Returns:
        None: Displays the plot or saves the figure to a file.
    """
    
    Xs, Ys, length_list = process_input(inputs)
        
    if np.mean(length_list) != np.max(length_list): # detect if scans have different length
        if pad_sequence == []:
            print('Different scan ranges, input pad_sequence to pad')
            return 
        else: # if pad sequence is provided, pad the shorter scans
            for i in range(len(Ys)):
                Ys[i] = np.pad(Ys[i], pad_sequence[i], mode='median')
            Xs = [Xs[np.argmax(length_list)]]*len(Ys)
            Xs = [Xs[np.argmax(length_list)]]*len(Ys)
        
    if fig == None and ax == None:
        fig, ax = plt.subplots(figsize=figsize)
    elif (fig == None and ax != None) or (fig != None and ax == None):
        raise ValueError('fig and ax should be provided together for customized plot')
     
    for i, (X, Y) in enumerate(zip(Xs, Ys)):
        Y[Y==0] = 1  # remove all 0 value
        if diff:
            Y = Y * diff**(len(Ys)-i-1)
        if legend_style == 'legend':
            ax.plot(X, Y, label=labels[i], color=colors[i])
        elif legend_style == 'label':
            line, = ax.plot(X, Y)
            if text_offset_ratio != None:
                ax.text(X[-1]*text_offset_ratio[0], Y[-1]*text_offset_ratio[1], labels[i], fontsize=10, color=line.get_color())
            else:
                ax.text(X[-1], Y[-1], labels[i], fontsize=10, color=line.get_color())
        
    plt.yscale('log',base=10) 
    if isinstance(xrange, tuple):
        plt.xlim(xrange)  
        
    if isinstance(yrange, tuple):
        plt.ylim(yrange)  
        
    ax.set_xlabel(r"2$\Theta$ [°]")
    ax.set_ylabel('Intensity [a.u.]')
    if legend_style == 'legend':
        ax.legend()

    ax.set_title(title)
    if filename and fig==None and ax==None:
        plt.savefig(filename)
    elif filename and fig!=None and ax!=None:
        raise ValueError('Figure won\'t be saved when fig and ax are provided') 

    # plt.xticks(np.arange(*xrange, 1))
    ax.tick_params(axis="x", direction="in", top=True)
    ax.tick_params(axis="y", direction="in", right=True)    
    
    if grid: plt.grid()

    if fig == None and ax == None:
        plt.tight_layout()
        plt.show()