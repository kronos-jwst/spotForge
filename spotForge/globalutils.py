# Functions for fitting the global spot properties
#
# Created by: Adina Feinstein
#
# Last edited by: Adina Feinstein (05/19/25)

import jax
import numpy as np
import jax.numpy as jnp
from astropy import units
from astropy import constants
import matplotlib.pyplot as plt

__all__ = ['planck', 'flux_in_bandpass', 'rotating_projected_spot_area']

def assign_units(wvl, T):
    """ Assigns astropy units. """
    return wvl*units.nm, T*units.K

def planck(wvl, T):
    """
    Planck Blackbody function.

    Parameters
    ----------
    wvl : array
       Wavelength array (in nm).
    T : float
       Temperature (in K).
    """
    term1 = (2.0 * constants.h * constants.c**2.0) / wvl**5.0

    exp = (constants.h * constants.c) / (wvl * constants.k_B * T)
    term2 = 1.0 / (np.exp(exp) - 1.0)

    I = term1 * term2
    I = I.to(units.erg/units.s/units.cm**2/units.AA)
    return I

def flux_in_bandpass(wvl, T, bandfunc):
    """
    Integrate over the Planck function over a bandpass function.

    Parameters
    ----------
    wvl : array
       Wavelength array (in nm).
    T : float
       Temperature (in K).
    bandfunc : array
       The response function for a given bandpass.
    """
    wvl, T = assign_units(wvl, T)
    I = planck(wvl, T)
    return np.trapz(I * bandfunc, wvl)


def rotating_projected_spot_area(time, P_rot, inc, T_phot, params):
    """
    Returns time-dependent stellar parameters base on a rotating star with
    changing spot properties.

    Parameters
    ----------
    time : jnp.array
       Observation times (in units of days).
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

    Returns
    -------
    T_eff_total : jnp.array
       Calculated effective temperature as a function of the stellar parameters.
       Combines the temperature of the photosphere and all of the spots.
    A_spot : jnp.array
       Calculated fractional projected area of the spots.
    """
    lon = jnp.array(params['lon'])
    lat = jnp.array(params['lat'])
    rad = jnp.array(params['spot_rad'])
    delta_T = jnp.array(params['delta_T'])

    T_spot_indiv = T_phot - delta_T

    def spot_area_i(t):
        lon_i = lon + 2.0 * jnp.pi * (t / P_rot)
        mu = jnp.sin(lat) * jnp.cos(inc) + jnp.cos(lat) * jnp.sin(inc) * jnp.cos(lon_i)

        mu_visible = jnp.where(mu > 0.0, mu, 0.0)
        spot_vis = rad**2.0 * mu_visible

        A = jnp.sum(spot_vis)

        T_spot = jnp.sum((T_spot_indiv * spot_vis)**4.0)
        return A, T_spot

    A_spot, T_eff_spot = jax.vmap(spot_area_i)(time)
    T_eff_total = ((1.0 - A_spot) * T_phot**4.0 + T_eff_spot)**0.25

    return A_spot, T_eff_total
