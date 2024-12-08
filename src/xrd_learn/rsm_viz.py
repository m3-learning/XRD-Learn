"""
This module provides a function for visualizing X-ray diffraction (XRD) data, including reciprocal space maps (RSMs). These visualizations are useful for analyzing the structural properties of 
materials through XRD experiments.

Functions:
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

__author__ = "Joshua C. Agar, Yichen Guo"
__copyright__ = "Joshua C. Agar, Yichen Guo"
__license__ = "MIT"

  
class RSMPlotter:
    def __init__(self, plot_params=None):
        """
        Initialize the RSMPlotter with optional plot parameters.
        
        Args:
            plot_params (dict): Dictionary of plot parameters.
        """
        default_params = {
                        "reciprocal_space": True,
                        'title': None,
                        'figsize': None, 
                        "cmap": plt.cm.viridis,
                        "title_fontsize": 12,
                        "label_fontsize": 10,
                        "tick_fontsize": 8,
                        "log_scale": True,
                        "cbar_value_format": 'actual',
                        "cbar_levels": 20,
                        "cbar_ticks": 10,
                        "cbar_size": 8, 
                        "cbar_fraction": 0.05,
                        "cbar_pad":  0.02,
                        'show_xaxis': 'last',
                        'show_yaxis': 'first',
                        "vmin": 3,
                        "vmax": 1000,
                        'custom_bg_color': None,
                        'save_path': None,
                        }

        # Merge default_params with any provided plot_params, prioritizing plot_params values
        self.plot_params = {**default_params, **(plot_params or {})}

        
    def plot(self, file, ax=None, figsize=None, cbar_ax=None, ignore_yaxis=False):
        """
        Plot the reciprocal space map (RSM) or direct space map.

        Args:
            file (str): Path to the XRDML file.
            axes (list, optional): List of axes for the plot.
            ax (matplotlib.axes.Axes, optional): Custom axes for the plot.
            figsize (tuple, optional): Figure size. Default is None.

        Returns:
            tuple: Qx, Qz, and intensity arrays.
        """
        # Extract data from the XRDML file
        curve_shape = xu.io.getxrdml_scan(file)[0].shape
        omega, two_theta, intensity = xu.io.panalytical_xml.getxrdml_map(file)
        omega = omega.reshape(curve_shape)
        two_theta = two_theta.reshape(curve_shape)
        intensity = intensity.reshape(curve_shape)
        
        fig, ax = self._prepare_figure(ax, figsize)
        
        reciprocal_space = self.plot_params.get("reciprocal_space", True)
        if reciprocal_space:
            Qx, Qz = self._calculate_reciprocal_space(omega, two_theta)
            cs = self._plot_reciprocal_space(ax, Qx, Qz, intensity)
        else:
            self._plot_direct_space(ax, omega, two_theta, intensity)
            
        if cbar_ax:
            cbar_ticks = self.plot_params.get("cbar_ticks", 10)
            cbar_size = self.plot_params.get("cbar_size", 10)
            self._add_colorbar(fig, cbar_ax, cs, cbar_ticks, cbar_size)
            
        self._apply_plot_settings(ax,ignore_yaxis )

        save_path = self.plot_params.get("save_path")
        if save_path:
            plt.savefig(f"{save_path}.svg", dpi=600)
            plt.savefig(f"{save_path}.png", dpi=600)

        if fig == None and ax == None:
            plt.tight_layout()
            plt.show()

        return Qx, Qz, intensity

    def _prepare_figure(self, ax, figsize):
        """Prepare the figure and axes for plotting."""

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        return fig, ax
    
    def _get_subplot_indices(self, fig, ax):
        """Get the row and column index of the given axis."""
        axes = fig.axes  # List of all axes in the figure
        ncols = len(set(a.get_position().x0 for a in axes))  # Number of columns

        # Find the current axis index
        ax_index = axes.index(ax)
        nrows = len(axes) // ncols  # Number of rows

        # Calculate row and column index
        row_index = ax_index // ncols
        col_index = ax_index % ncols

        return row_index, col_index, nrows, ncols

    def _calculate_reciprocal_space(self, omega, two_theta):
        """Calculate Qx and Qz from omega and two-theta values."""
        wavelength = 1.54  # Angstrom
        k = 2 * np.pi / wavelength
        Qz = k * (np.sin(np.deg2rad(two_theta - omega)) + np.sin(np.deg2rad(omega)))
        Qx = k * (np.cos(np.deg2rad(omega)) - np.cos(np.deg2rad(two_theta - omega)))
        return Qx, Qz

    def _plot_reciprocal_space(self, ax, Qx, Qz, intensity):
        """
        Plot the reciprocal space map with optional logarithmic scaling.
        
        Args:
            fig: Figure object
            ax: Axis object
            Qx: 2D array of Qx values
            Qz: 2D array of Qz values
            intensity: 2D array of intensity values
            cbar_ax: Axis object for the colorbar, note that:
                    'last' will add colorbar to the last column only, 
                    'every' will add colorbar to every plot, 
                    ax will add colorbar to the current plot only,
                    None will not add colorbar at all.
                    Nested axes is only compatible with ax and None option.
        """
        
        log_scale = self.plot_params.get("log_scale", True)
        cmap = self.plot_params.get("cmap", plt.cm.viridis)
        vmin, vmax = self._get_intensity_limits(intensity)
        cbar_levels = self.plot_params.get("cbar_levels", 20)
        custom_bg_color = self.plot_params.get("custom_bg_color")

        from scipy.ndimage import zoom
        downsample_factor = 0.5
        Qx = zoom(Qx, downsample_factor)
        Qz = zoom(Qz, downsample_factor)
        intensity = zoom(intensity, downsample_factor)

        if log_scale:
            intensity = self._adjust_intensity(intensity, vmin, vmax)
            levels = np.logspace(np.log10(vmin), np.log10(vmax), cbar_levels)
            cmap = self._create_custom_colormap(cmap, custom_bg_color)
            cs = ax.contourf(Qx, Qz, intensity, levels=levels, cmap=cmap,
                             norm=colors.LogNorm(vmin=vmin, vmax=vmax), extend='neither')
            cs.set_rasterized(True) # improve the render quality but get a warning
        else:
            cs = ax.contourf(Qx, Qz, intensity, levels=cbar_levels, cmap=cmap)
            
        # Add colorbar only if this is the far-right column (exclude the last column for the colorbar)
        # if cbar_ax == 'last':
        #     # Detect layout information
        #     row_index, col_index, nrows, ncols = self._get_subplot_indices(fig, ax)
        #     if col_index == ncols - 1: # Add colorbar to the last column only
        #         self._add_colorbar(fig, ax, cs, cbar_ticks, cbar_size)
        #         # cbar = fig.colorbar(cs, ax=fig.axes, orientation='vertical', 
        #         #                     fraction=self.plot_params.get('cbar_fraction', 0.1), 
        #         #                     pad=self.plot_params.get('cbar_pad', 0.1))
        # elif cbar_ax == 'every':
        #     row_index, col_index, nrows, ncols = self._get_subplot_indices(fig, ax)
        #     if col_index//2 == 1:
        #         self._add_colorbar(fig, ax, cs, cbar_ticks, cbar_size)
                
        # elif isinstance(cbar_ax, plt.Axes):
            # self._add_colorbar(fig, ax, cs, cbar_ticks, cbar_size)
            

        # # Only show y-axis for the first plot
        # if self.plot_params.get("show_yaxis", 'all') == 'first' and col_index != 0:
        #     ax.set_yticks([])
        #     ax.set_yticklabels([])
        #     ax.set_ylabel('')
        # elif self.plot_params.get("show_yaxis", 'all') == 'last':
        #     raise ValueError('show_yaxis="last" is not supported for RSM plots.')
            
        # # Only show x-axis for the last plot
        # elif self.plot_params.get("show_xaxis", 'all') == 'first':
        #     raise ValueError('show_xaxis="first" is not supported for RSM plots.')
        # elif self.plot_params.get("show_xaxis", 'all') == 'last' and row_index != nrows - 1:
        #     ax.set_xticks([])
        #     ax.set_xticklabels([])
        #     ax.set_xlabel('')
            
        self._blend_background_color(intensity, Qx, Qz, ax, cs, custom_bg_color)
        return cs

    def _add_colorbar(self, fig, cax, cs, cbar_ticks, cbar_size):
        """
        Add a colorbar with custom tick formatting.
        
        Args:
            fig: Figure object
            cax: Axis object to attach the colorbar
            cs: ContourSet object
            cbar_ticks: Number of ticks on the colorbar
            custom_bg_color: Custom background color for the colormap
        """
        # cbar = plt.colorbar(cs, ax=ax, extendfrac='auto')
        # This code snippet is adding a colorbar to the plot. Here's a breakdown of what each line is
        # doing:
        cbar = fig.colorbar(cs, cax=cax, orientation='vertical', 
                            fraction=self.plot_params.get('cbar_fraction', 0.1), pad=self.plot_params.get('cbar_pad', 0.1))
        # cbar = fig.colorbar(cs, ax=axes, orientation='vertical', 
        #                     fraction=self.plot_params.get('cbar_fraction', 0.1), pad=self.plot_params.get('cbar_pad', 0.1))

        # Create custom tick locations
        tick_locations = np.logspace(np.log10(self.plot_params.get("vmin", 3)), np.log10(self.plot_params.get("vmax", 3000)), num=cbar_ticks)
        cbar.set_ticks(tick_locations)
        cbar.ax.tick_params(labelsize=cbar_size)  # Set tick label size

        def format_func(value, tick_number):
            if self.plot_params.get("cbar_value_format", 'actual') == 'log':
                return f"$10^{{{np.log10(value):.0f}}}$"
            elif self.plot_params.get("cbar_value_format", 'actual') == 'actual':
                return f"{value:.0f}"
        
        cbar.formatter = ticker.FuncFormatter(format_func)
        cbar.update_ticks()
        
        
    def _blend_background_color(self, intensity, Qx, Qz, ax, cs, custom_bg_color):
        '''
        Blend the background color with the canvas color for better effect.

        Args:
            intensity: 2D array of intensity values
            Qx: 2D array of Qx values
            Qz: 2D array of Qz values
            ax: Axis object
            cs: ContourSet object
            custom_bg_color: Custom background color for the colormap
        '''
        
        # draw a rectangle with the background color
        # Get the background value and its corresponding color
        if custom_bg_color == None:
            bg_value = np.bincount(intensity.flatten().astype(np.int32)).argmax()
            custom_bg_color = cs.cmap(cs.norm(bg_value))
        rect = Rectangle((np.min(Qx), np.min(Qz)), np.max(Qx)-np.min(Qx), np.max(Qz)-np.min(Qz), facecolor=custom_bg_color, edgecolor='none', zorder=-1)
        ax.add_patch(rect)


    def _plot_direct_space(self, ax, omega, two_theta, intensity):
        """Plot data in direct space (omega and two-theta)."""
        cs = ax.contourf(omega, two_theta, intensity,
                         levels=self.plot_params.get("cbar_levels", 20),
                         cmap=self.plot_params.get("cmap", plt.cm.viridis))
        ax.set_xlabel(r'$\omega$ [degree]', fontsize=self.plot_params.get("fontsize", 12))
        ax.set_ylabel(r'$2\theta$ [degree]', fontsize=self.plot_params.get("fontsize", 12))

    def _get_intensity_limits(self, intensity):
        """Determine intensity limits."""
        vmin = self.plot_params.get("vmin", intensity[intensity > 0].min())
        vmax = self.plot_params.get("vmax", intensity.max())
        return vmin, vmax

    def _adjust_intensity(self, intensity, vmin, vmax):
        """Clip intensity values to within specified limits."""
        intensity[intensity <= vmin] = vmin - 1e-10
        intensity[intensity >= vmax] = vmax - 1e-10
        return intensity

    def _create_custom_colormap(self, cmap, custom_bg_color):
        """Create a colormap with a custom background color if specified."""
        if custom_bg_color:
            color_list = cmap(np.linspace(0, 1, 256))
            color_list[0] = colors.to_rgba(custom_bg_color)
            return colors.LinearSegmentedColormap.from_list("custom", color_list)
        return cmap

    def _apply_plot_settings(self, ax, ignore_yaxis):
        """Apply axis limits, title, and tick settings."""
        if xlim := self.plot_params.get("xlim"):
            ax.set_xlim(xlim)
        if ylim := self.plot_params.get("ylim"):
            ax.set_ylim(ylim)

        ax.tick_params(axis="x", direction="in", top=True, labelsize=self.plot_params.get("tick_fontsize", 8))
        ax.tick_params(axis="y", direction="in", right=True, labelsize=self.plot_params.get("tick_fontsize", 8))    
                
        ax.set_xlabel(r'$Q_x$ [$\AA^{-1}$]', fontsize=self.plot_params['label_fontsize'], fontweight='bold')
        ax.set_ylabel(r'$Q_z$ [$\AA^{-1}$]', fontsize=self.plot_params['label_fontsize'], fontweight='bold')

        if ignore_yaxis:
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylabel('')
            
        if title := self.plot_params.get("title"):
            ax.set_title(title, fontsize=self.plot_params.get("title_fontsize", 12))