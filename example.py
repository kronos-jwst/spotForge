import numpy as np
import jax.numpy as jnp
from astropy import units as u
import matplotlib.pyplot as plt

from spotForge import SpotForge, globalutils


def sine_wave(t, per, ampl):
    return ampl * np.sin(1.0/per.value * t.value)

# Synthetic light light curves
time = np.linspace(0, 30, 100) * u.day
per = 5.1 * u.day
g = sine_wave(time, per, 2.0)
r = sine_wave(time, per, 1.8)
i = sine_wave(time, per, 1.0)
flux_err = np.full(len(time), 0.05)

light_curves = {
    'g': (time, g, flux_err),
    'r': (time, r, flux_err),
    'i': (time, i, flux_err)
}


bandpasses = {
    'g': 'sdss',
    'r': 'sdss',
    'i': 'johnson'
}

sf = SpotForge(light_curves, transit_curve=None, bandpasses=bandpasses)
mod, resid = sf.global_model(5600, 4000, 0.3)

plt.plot(time, g)
plt.plot(time, mod['g'])
