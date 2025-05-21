# Visualization tools for the outputs of spotForge
import numpy as np
from astropy import units
import matplotlib.pyplot as plt

__all__ = ['plot_light_curves', 'plot_responses', 'project_spot_on_mollweide',
           'plot_spot_mollweide']

def define_color(wvl, wlen):
    return

def plot_light_curves(light_curves, responses):
    """
    Plots the multi-color photometric light curves on top of each other.
    """

    for key in list(light_curves.keys()):
        plt.errorbar(light_curves[key][0], light_curves[key][1],
                     yerr=light_curves[key][2], marker='.', linestyle='')

    plt.show()

    return

def plot_responses(responses):
    """
    Plots the response functions for each photometric bandpass.
    """

    for key in list(responses.keys()):

        plt.fill_between(responses[key]['wvl'], np.zeros(len(responses[key]['wvl'])),
                         responses[key]['resp'], alpha=0.3, zorder=0)
        plt.plot(responses[key]['wvl'],
                 responses[key]['resp'], lw=2.5)

    return

def project_spot_on_mollweide(lat0, lon0, radius, contrast, resolution=300):
    """
    Project a circular starspot onto a Mollweide projection.

    Parameters
    ----------
    lat0 : array
       Latitude of the spot (in radians).
    lon0 : array
       Longitude of the spot (in radians).
    radius : array
       Angular radius of the spot (in radians).
    contrast : array
       Contrast of the spot with respect to the photosphere.
    resolution: int, optional
       The number of pixels per axis (lat × lon). Default is 300.

    Returns
    -------
    lat_grid : array
       Array of latitude grid values, prepared for Mollweide projection.
    lon_grid : array
       Array of longitude grid values, prepared for Mollweide projection.
    spot_mask: array
       Array of where the spots are, prepared for Mollweide projection.
    """
    # Define lat/lon grid in radians
    lat = np.linspace(-np.pi/2, np.pi/2, resolution)
    lon = np.linspace(-np.pi, np.pi, resolution)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    for i in range(len(lat0)):
        # Compute angular distance from spot center using spherical law of cosines
        cos_dtheta = (
            np.sin(lat0[i]) * np.sin(lat_grid) +
            np.cos(lat0[i]) * np.cos(lat_grid) * np.cos(lon_grid - lon0[i])
        )
        dtheta = np.arccos(np.clip(cos_dtheta, -1.0, 1.0))

        # Mask: 1 inside the spot radius, 0 outside
        if i == 0:
            spot_mask = (dtheta <= radius[i]).astype(float)
        else:
            spot_mask += (dtheta <= radius[i]).astype(float)


    return lon_grid, lat_grid, spot_mask

def plot_spot_mollweide(lon_grid, lat_grid, spot_mask, title='Projected Spot'):
    """Plot the spot mask on a Mollweide projection."""
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='mollweide')
    pcm = ax.pcolormesh(-lon_grid, lat_grid, spot_mask, shading='auto', cmap='plasma')
    ax.grid(True)
    ax.set_title(title)
    plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.07, label='Spot Visibility')
    plt.show()
    return
