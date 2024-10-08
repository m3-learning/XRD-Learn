"""
This module provides functions for visualizing X-ray diffraction (XRD) data, including plotting XRD scans and 
reciprocal space maps (RSMs). These visualizations are useful for analyzing the structural properties of 
materials through XRD experiments.

Functions:
    - plot_xrd: Plots XRD scans for different datasets, allowing for customizations such as padding sequences, 
      logarithmic scales, and figure saving options.
    - plot_rsm: Generates reciprocal space maps (RSMs) from XRDML files, with support for logarithmic scaling, 
      color customization, and contour plotting in reciprocal or angular space.

References:
    - https://matplotlib.org/stable/tutorials/index.html
    - https://xrayutilities.readthedocs.io/en/latest/
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors
import xrayutilities as xu
from matplotlib.patches import Rectangle
from .xrd_utils import process_input

__author__ = "Joshua C. Agar, Yichen Guo"
__copyright__ = "Joshua C. Agar, Yichen Guo"
__license__ = "MIT"


def plot_xrd(inputs, labels, title='XRD Scan', xrange=None, diff=1e3, fig=None, ax=None, figsize=(6,4), 
             grid=False, pad_sequence=[], filename=None):
    
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
        grid (bool, optional): Whether to show gridlines. Default is False.
        pad_sequence (list, optional): Sequence for padding datasets if they have different ranges.
        filename (str, optional): If provided, the plot will be saved with the given filename.
    
    Returns:
        None: Displays the plot or saves the figure to a file.
    """
    
    Xs, Ys, length_list = process_input(inputs)
        
    if np.mean(length_list) != np.max(length_list):
        if pad_sequence == []:
            print('Different scan ranges, input pad_sequence to pad')
            return 
        else:
            for i in range(len(Ys)):
                Ys[i] = np.pad(Ys[i], pad_sequence[i], mode='median')
    X = Xs[np.argmax(length_list)]
        
    if fig == None and ax == None:
        fig, ax = plt.subplots(figsize=figsize)
    elif (fig == None and ax != None) or (fig != None and ax == None):
        raise ValueError('fig and ax should be provided together for customized plot')
     
    for i, Y in enumerate(Ys):
        Y[Y==0] = 1  # remove all 0 value
        if diff:
            Y = Y * diff**(len(Ys)-i-1)
        ax.plot(X, Y, label=labels[i])
        
    plt.yscale('log',base=10) 
    if isinstance(xrange, tuple):
        plt.xlim(xrange)  
        
    ax.set_xlabel(r"2$\Theta$")
    ax.set_ylabel('Intensity [a.u.]')
    ax.legend()

    ax.set_title(title)
    if filename and fig==None and ax==None:
        plt.savefig(filename)
    elif filename and fig!=None and ax!=None:
        raise ValueError('Figure won\'t be saved when fig and ax are provided') 

    # plt.xticks(np.arange(*xrange, 1))
    if grid: plt.grid()

    if fig == None and ax == None:
        plt.tight_layout()
        plt.show()


def plot_rsm(file, reciprocal_space=True, title=None, cmap=cm.viridis, xlim=None, ylim=None, fig=None, ax=None, figsize=(6,4),
             log_scale=True, cbar_value_format='log',cbar_levels=20, cbar_ticks=10, vmin=None, vmax=None, 
             custom_bg_color=None, save_path=None):
    
    """Plot reciprocal space maps (RSM) from XRDML files.

    This function processes XRDML files and generates a reciprocal space map, with options
    for logarithmic scaling, color customization, and saving the plot.

    Args:
        file (str): Path to the XRDML file to be processed.
        reciprocal_space (bool, optional): Whether to plot in reciprocal space. Default is True.
        title (str, optional): Title of the plot. Default is None.
        cmap (matplotlib.colors.Colormap, optional): Colormap to be used. Default is cm.viridis.
        xlim (tuple, optional): Limits for the x-axis.
        ylim (tuple, optional): Limits for the y-axis.
        fig (matplotlib.figure.Figure, optional): Custom figure for the plot.
        ax (matplotlib.axes.Axes, optional): Custom axes for the plot.
        figsize (tuple, optional): Figure size. Default is (6, 4).
        log_scale (bool, optional): Whether to apply logarithmic scaling to the intensity values. Default is True.
        cbar_value_format (str, optional): Format for colorbar values, either 'log' or 'actual'. Default is 'log'.
        cbar_levels (int, optional): Number of levels for the colorbar. Default is 20.
        cbar_ticks (int, optional): Number of ticks for the colorbar. Default is 10.
        vmin (float, optional): Minimum value for the intensity scaling.
        vmax (float, optional): Maximum value for the intensity scaling.
        custom_bg_color (str, optional): Background color for values below vmin. Default is None.
        save_path (str, optional): Path to save the plot. If provided, the plot will be saved.
    
    Returns:
        tuple: Qx, Qz, and intensity arrays.
    """
    
    
    # print(np.zeros((2,2)))
    curve_shape = xu.io.getxrdml_scan(file)[0].shape
    omega, two_theta, intensity = xu.io.panalytical_xml.getxrdml_map(file)

    omega = omega.reshape(curve_shape)
    two_theta = two_theta.reshape(curve_shape)
    intensity = intensity.reshape(curve_shape)
    
    # if log_scale: # remove all 0 value
        # intensity[intensity==0]=1
    if fig == None and ax == None:
        fig, ax = plt.subplots(figsize=figsize)
    elif (fig == None and ax != None) or (fig != None and ax == None):
        raise ValueError('fig and ax should be provided together for customized plot')
    
    if reciprocal_space:
        wavelength = 1.54  # unit: angstrom
        k = 2 * np.pi / wavelength
        Qz = k * (np.sin(np.deg2rad(two_theta-omega)) + np.sin(np.deg2rad(omega)))
        Qx = k * (np.cos(np.deg2rad(omega)) - np.cos(np.deg2rad(two_theta-omega)))

        if log_scale:
            intensity_filtered = intensity[intensity > 0]
            if vmin == None:
                vmin = intensity_filtered.min()
            if vmax == None:
                vmax = intensity_filtered.max()

            intensity[intensity <= vmin] = vmin-1e-10
            intensity[intensity >= vmax] = vmax-1e-10

            levels = np.logspace(np.log10(vmin), np.log10(vmax), num=cbar_levels)

            # Create a custom colormap with the arbitrary color for the minimum value
            if custom_bg_color != None:
                cmap_orig = plt.get_cmap(cmap)
                color_list = cmap_orig(np.linspace(0, 1, 256))
                color_list[0] = colors.mcolors.to_rgba(custom_bg_color)  # Set the first color to your arbitrary color
                custom_cmap = colors.mcolors.LinearSegmentedColormap.from_list("custom", color_list)
            else:
                custom_cmap = cmap

            # Use the custom colormap in contourf
            cs = ax.contourf(Qx, Qz, intensity, levels=levels, cmap=custom_cmap, 
                            norm=colors.LogNorm(vmin=vmin, vmax=vmax), extend='neither')
            cbar = plt.colorbar(cs, ax=ax, extendfrac='auto')

            # draw a rectangle with the background color
            # Get the background value and its corresponding color
            if custom_bg_color == None:
                bg_value = np.bincount(intensity.flatten().astype(np.int32)).argmax()
                custom_bg_color = cs.cmap(cs.norm(bg_value))
            rect = Rectangle((np.min(Qx), np.min(Qz)), np.max(Qx)-np.min(Qx), np.max(Qz)-np.min(Qz), facecolor=custom_bg_color, edgecolor='none', zorder=-1)
            ax.add_patch(rect)

            # Create custom tick locations
            tick_locations = np.logspace(np.log10(vmin), np.log10(vmax), num=cbar_ticks)
            cbar.set_ticks(tick_locations)
            
            def format_func(value, tick_number):
                if cbar_value_format == 'log':
                    return f"$10^{{{np.log10(value):.1f}}}$"
                elif cbar_value_format == 'actual':
                    return f"{value:.1f}"
            
            cbar.formatter = ticker.FuncFormatter(format_func)
            cbar.update_ticks()
    
        else:
            cs = ax.contourf(Qx, Qz, intensity, levels=cbar_levels, cmap=cmap)
            cbar = plt.colorbar(cs)
        
        ax.set_xlabel(r'$Q_x$ [$\AA^{-1}$]')
        ax.set_ylabel(r'$Q_z$ [$\AA^{-1}$]')
        cbar.set_label('Intensity')

    else:
        cs = ax.contourf(omega, two_theta, intensity, n_levels=cbar_levels,
                        locator=ticker.LogLocator(), cmap=cmap, norm=colors.LogNorm())
        ax.set_xlabel(r'$\omega$ [degree]')
        ax.set_ylabel(r'$2\theta$ [degree]')
        if log_scale:
            ax.set_yscale('log')
        cbar = fig.colorbar(cs, ax=ax)
        Qx, Qz = omega, two_theta

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # ax.tick_params(axis="both", direction="in", color='white')
    # ax.tick_params(axis="x", top=True)
    # ax.tick_params(axis="y", right=True)
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")

    if title:
        ax.set_title(title)

    if save_path:
        plt.savefig(save_path+'.svg', dpi=600)
        plt.savefig(save_path+'.png', dpi=600)

    if fig == None and ax == None:
        plt.tight_layout()
        plt.show()

    return Qx, Qz, intensity