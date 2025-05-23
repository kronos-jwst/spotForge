import numpy as np
import speclite.filters

#from .globalutils import global_flux_model
from .plotting import plot_light_curves

__all__ = ['SpotForge']


class SpotForge(object):
    def __init__(self, light_curves, transit_curve, bandpasses):
        """
        A class to simultaneously fit for the spot properties of a given star
        with a set of photometric light curves and a given transit light curve
        with starspot crossing events.

        Parameters
        ----------
        light_curves : dict
           {band: (time, flux, flux_err)}
        transit_curve : dict
           {'time': time, 'flux': flux, 'flux_err':flux_err}. Currently works for
           a single bandpass for transit observations.
        bandpasses : dict
           Response function for each bandpass in `light_curves`.
        """
        self.light_curves = light_curves
        self.transit_curve = transit_curve
        self.bandpasses = bandpasses

        self.get_response_func()


    def set_priors(self, **priors):
        """ Initializes priors for the local and global fits. """
        self.priors = priors


    def global_model(self, P_rot, inc, T_phot, params):
        """
        Fits the global spot temperature and fraction coverage from multi-color
        photometric light curves.

        Parameters
        ----------
        P_rot : float
           The measured rotation period of the star (in units of days).
        inc : float
           The measured inclination of the star (in units of degrees).
        T_phot : float
           The temperature of the photosphere (in units of K).
        params : dict
           Dictionary of stellar properties. This dictionary must include:
           - 'delta_T' : array, difference in temperature between the photosphere and
                               the spots
           - 'lon' : array, initial longitude (radians).
           - 'lat' : array, initial latitude (radians)
           - 'spot_rad' : array, angular radius of the spot (radians)
        """
