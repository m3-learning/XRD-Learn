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
        self.plot_params = plot_params or {}
        
    def plot(self, file, fig=None, ax=None, figsize=None):
        """
        Plot the reciprocal space map (RSM) or direct space map.

        Args:
            file (str): Path to the XRDML file.

        Returns:
            tuple: Qx, Qz, and intensity arrays.
        """
        # Extract data from the XRDML file
        curve_shape = xu.io.getxrdml_scan(file)[0].shape
        omega, two_theta, intensity = xu.io.panalytical_xml.getxrdml_map(file)
        omega = omega.reshape(curve_shape)
        two_theta = two_theta.reshape(curve_shape)
        intensity = intensity.reshape(curve_shape)
        
        fig, ax = self._prepare_figure(fig, ax, figsize)
        
        reciprocal_space = self.plot_params.get("reciprocal_space", True)
        if reciprocal_space:
            Qx, Qz = self._calculate_reciprocal_space(omega, two_theta)
            self._plot_reciprocal_space(fig, ax, Qx, Qz, intensity)
        else:
            self._plot_direct_space(ax, omega, two_theta, intensity)
        self._apply_plot_settings(ax)
        
        ax.tick_params(axis="x", direction="in", top=True)
        ax.tick_params(axis="y", direction="in", right=True)    
    
        save_path = self.plot_params.get("save_path")
        if save_path:
            plt.savefig(f"{save_path}.svg", dpi=600)
            plt.savefig(f"{save_path}.png", dpi=600)

        if fig == None and ax == None:
            plt.tight_layout()
            plt.show()

        return Qx, Qz, intensity

    def _prepare_figure(self, fig, ax, figsize):
        """Prepare the figure and axes for plotting."""

        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        elif (fig is None) != (ax is None):
            raise ValueError('Both "fig" and "ax" should be provided together.')

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


    def _plot_reciprocal_space(self, fig, ax, Qx, Qz, intensity):
        """Plot the reciprocal space map with optional logarithmic scaling."""
        
        log_scale = self.plot_params.get("log_scale", True)
        cmap = self.plot_params.get("cmap", plt.cm.viridis)
        vmin, vmax = self._get_intensity_limits(intensity)
        cbar_levels = self.plot_params.get("cbar_levels", 20)
        cbar_ticks = self.plot_params.get("cbar_ticks", 10)
        custom_bg_color = self.plot_params.get("custom_bg_color")

        # Detect layout information
        row_index, col_index, nrows, ncols = self._get_subplot_indices(fig, ax)

        if log_scale:
            intensity = self._adjust_intensity(intensity, vmin, vmax)
            levels = np.logspace(np.log10(vmin), np.log10(vmax), cbar_levels)
            cmap = self._create_custom_colormap(cmap, custom_bg_color)
            cs = ax.contourf(Qx, Qz, intensity, levels=levels, cmap=cmap,
                             norm=colors.LogNorm(vmin=vmin, vmax=vmax), extend='neither')
        else:
            cs = ax.contourf(Qx, Qz, intensity, levels=cbar_levels, cmap=cmap)
            
        # Add colorbar only if this is the far-right column
        if col_index == ncols - 1:
            self._add_colorbar(fig, fig.axes, cs, cbar_ticks)
            # cbar = fig.colorbar(cs, ax=fig.axes, orientation='vertical', 
            #                     fraction=self.plot_params.get('cbar_fraction', 0.1), 
            #                     pad=self.plot_params.get('cbar_pad', 0.1))
                
        ax.set_xlabel(r'$Q_x$ [$\AA^{-1}$]', fontsize=self.plot_params.get("fontsize", 12))
        ax.set_ylabel(r'$Q_z$ [$\AA^{-1}$]', fontsize=self.plot_params.get("fontsize", 12))

        # Only show y-axis for the first plot
        if self.plot_params.get("show_yaxis", 'all') == 'first' and col_index != 0:
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylabel('')
        elif self.plot_params.get("show_yaxis", 'all') == 'last':
            raise ValueError('show_yaxis="last" is not supported for RSM plots.')
            
        # Only show x-axis for the last plot
        elif self.plot_params.get("show_xaxis", 'all') == 'first':
            raise ValueError('show_xaxis="first" is not supported for RSM plots.')
        elif self.plot_params.get("show_xaxis", 'all') == 'last' and row_index != nrows - 1:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlabel('')
            
        self._blend_background_color(intensity, Qx, Qz, ax, cs, custom_bg_color)
            

    def _add_colorbar(self, fig, ax, cs, cbar_ticks):
        """Add a colorbar with custom tick formatting."""
        # cbar = plt.colorbar(cs, ax=ax, extendfrac='auto')
        cbar = fig.colorbar(cs, ax=ax, orientation='vertical', 
                            fraction=self.plot_params.get('cbar_fraction', 0.1), pad=self.plot_params.get('cbar_pad', 0.1))


        # Create custom tick locations
        tick_locations = np.logspace(np.log10(self.plot_params.get("vmin", 3)), np.log10(self.plot_params.get("vmax", 3000)), num=cbar_ticks)
        cbar.set_ticks(tick_locations)
        
        def format_func(value, tick_number):
            if self.plot_params.get("cbar_value_format", 'actual') == 'log':
                return f"$10^{{{np.log10(value):.1f}}}$"
            elif self.plot_params.get("cbar_value_format", 'actual') == 'actual':
                return f"{value:.1f}"
        
        cbar.formatter = ticker.FuncFormatter(format_func)
        cbar.update_ticks()
        
        
    def _blend_background_color(self, intensity, Qx, Qz, ax, cs, custom_bg_color):
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

    def _apply_plot_settings(self, ax):
        """Apply axis limits, title, and tick settings."""
        if xlim := self.plot_params.get("xlim"):
            ax.set_xlim(xlim)
        if ylim := self.plot_params.get("ylim"):
            ax.set_ylim(ylim)

        ax.tick_params(axis="x", labelsize=self.plot_params.get("fontsize", 12))
        ax.tick_params(axis="y", labelsize=self.plot_params.get("fontsize", 12))

        if title := self.plot_params.get("title"):
            ax.set_title(title, fontsize=self.plot_params.get("fontsize", 12))