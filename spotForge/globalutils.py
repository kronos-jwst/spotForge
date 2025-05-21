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

__all__ = ['planck', 'compute_normalized_amplitude',
           'rotating_projected_spot_area', 'generate_light_curve',
           'build_param_names', 'build_bounds', 'log_posterior_multi',
           'log_likelihood_multi', 'compute_spot_contrast']

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
    wvl = wvl.to(units.cm).value
    h = constants.h.cgs.value
    c = constants.c.cgs.value
    kB = constants.k_B.cgs.value

    term1 = (2.0 * h * c**2.0) / wvl**5.0 # erg / s / cm^3

    exp = (h * c) / (wvl * kB * T) # unitless
    term2 = 1.0 / (jnp.exp(exp) - 1.0)

    I = term1 * term2 * 1e-8 # erg / s / cm^2 / AA
    return I


def compute_spot_contrast(T_phot, delta_T, wvl):
    """
    Computes the spot contrast with respect to the photosphere.

    Parameters
    ----------
    T_phot : float
       Photospheric temperature (in K).
    delta_T : float
       Difference in temperature between the photosphere and the spot (in K).
    wvl : array
       Array of wavelengths to integrate over (in nm).
    """
    T_spot = T_phot - delta_T

    I_star = planck(wvl, T_phot)
    I_spot = planck(wvl, T_spot)

    return jnp.trapezoid(I_spot, wvl) / jnp.trapezoid(I_star, wvl)

def compute_normalized_amplitude(light_curve):
    """
    Computes the normalized amplitude for a given light curve.
    """
    f_min = np.nanmin(light_curve)
    f_max = np.nanmax(light_curve)
    f_mean = np.nanmean(light_curve)
    return (f_max-f_min) / f_mean

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

def generate_light_curve(time, T_eff, wvl, rf):
    """
    Generates a light curve for a star of a given effective temperature
    and rotation period in a given photometric bandpass.

    Parameters
    ----------
    time : jnp.array
       Observation times (in units of days).
    T_eff : jnp.array
       Effective temperature across the rotation period.
    bandpass : str
       The name of the bandpass to integrate the flux in.

    Returns
    -------
    flux : array
       Array of calculated flux values (in erg/s/cm^2/Angstrom)
    """
    T_eff = jnp.array(T_eff)

    wvl = jnp.array(wvl) * units.nm
    rf  = jnp.array(rf)

    def flux_over_time(T):
        I = planck(wvl, T)
        return jnp.trapezoid(I * rf, wvl)

    flux = jax.vmap(flux_over_time)(T_eff)
    return flux#/jnp.nanmedian(flux)

def generate_model(time, P_rot, inc, T_phot, params, wvl, rf):
    """
    Generates the model light curve for a given set of spot properties and
    photometric bandpass.

    Parameters
    ----------
    """

    a_spot, t_eff_total = rotating_projected_spot_area(time,
                                                       P_rot,
                                                       inc,
                                                       T_phot,
                                                       params)

    flux = generate_light_curve(time, teff, wvl, rf)
    return flux

def build_param_names(n_spots=1):
    """
    Defines the parameter names for n number of spots.

    Parameters
    ----------
    n_spots : int, optional
       The number of spots to model. Default is 1.

    Returns
    -------
    names : list
       List of parameter names for the `emcee` model.
    """
    names = ['T_phot']
    for i in range(n_spots):
        names += [f"delta_T_{i}", f"lat_{i}", f"lon_{i}", f"radius_{i}"]
    return names

def build_bounds(n_spots=1, T_phot_bounds=[4000, 6000],
                 delta_T_bounds=[10, 1000], radius_bounds=(0.001, 0.5)):
    """
    Creates the bounds for each parameter in the `emcee` fit.

    Parameters
    ----------
    n_spots : int, optional
       The number of spots to model. Default is 1.
    T_phot_bounds : tuple, optional
       The lower and upper bounds to place on the photospheric temperature.
       Default is (4000, 6000).
    delta_T_bounds : tuple, optional
       The lower and upper bounds to place on the difference in temperature
       between the spots and the photosphere. Default is (10, 1000). In theory,
       one could use negative values to fit for facular regions, although this
       hasn't been tested.
    radius_bounds : tuple, optional
       The lower and upper bounds to place on the radius of any given spot.
       Default is (0.001, 0.5).
    """
    bounds = {'T_phot': T_phot_bounds}

    for i in range(n_spots):
        bounds[f'delta_T_{i}'] = delta_T_bounds
        bounds[f'lon_{i}'] = (0, 2.0*jnp.pi)
        bounds[f'lat_{i}'] = (-jnp.pi/2.0, jnp.pi/2.0)
        bounds[f'radius_{i}'] = radius_bounds

    return bounds

def unpack_params(theta, param_names):
    return dict(zip(param_names, theta))

def log_prior(theta, param_names, bounds):
    """
    Defines the log prior for the `emcee` fit.

    Parameters
    ----------
    theta : dict
       Initial values for each parameter.
    param_names : array
       The name of each parameter.
    bounds : dict
       Dictionary of lower/upper bounds for each parameter.
    """
    for val, name in zip(theta, param_names):
        lo, hi = bounds[name]
        if not (lo <= val <= hi):
            return -jnp.inf
    return 0.0

def log_likelihood_multi(theta, light_curves, param_names):
    """
    Defines a likelihood function for fitting multiple light curves across
    different photometric bandpasses.

    Parameters
    ----------
    theta : array
       Flat parameter vector.
    light_curves : list of dicts, each with:
       - 'time' : array
       - 'flux' : array
       - 'flux_err' : array
       - 'bandpass' : dict with 'wvl', 'rf'
    spot_model : function
       Time-dependent spot parameter generator.
    param_names : list
       Keys for the theta vector.

    Returns
    -------
    loglike : float
       Log likelihood value calculated for a given set of parameters.
    """
    params = unpack_params(theta, param_names)
    loglike = 0.0

    for lc in light_curves:
        time = jnp.array(lc['time'])
        flux_obs = jnp.array(lc['flux'])
        flux_err = jnp.array(lc['flux_err'])
        wvl, rf = jnp.array(lc['bandpass']['wvl']), jnp.array(lc['bandpass']['rf'])

        try:
            flux_model = generate_model(time, P_rot, inc, T_phot, params, wvl, rf)
            residuals = (flux_obs - flux_model) / flux_err
            loglike += -0.5 * jnp.sum(residuals**2.0 + jnp.log(2.0 * jnp.pi * flux_err**2.0))
        except Exception:
            return -jnp.inf

    return loglike

def log_posterior_multi(theta, light_curves, param_names, bounds):
    """
    Defines the posterior function for the `emcee` fit.

    Parameters
    ----------
    theta : dict
       Initial values for each parameter.
    light_curves : jnp.array
    spot_model
    param_names : array
       The name of each parameter.
    bounds : dict
       Dictionary of lower/upper bounds for each parameter.
    """
    lp = log_prior(theta, param_names, bounds)

    if not jnp.isfinite(lp):
        return -jnp.inf
    return lp + log_likelihood_multi(theta, light_curves, param_names)
