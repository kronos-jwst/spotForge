# Visualization tools for the outputs of spotForge
import numpy as np
from astropy import units
import matplotlib.pyplot as plt

__all__ = ['plot_light_curves', 'plot_responses']

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
