# A set of functions to return the wavelength and response functions for a given photometric bandpass
import jax.numpy as jnp
import speclite.filters
from astropy import units

__all__ = ['ResponseFunctions']


class ResponseFunctions(object):
    """
    Sets the attributes for the response functions. Currently works with
    SDSS and Johnson/Cousins filters only.
    """
    def __init__(self):

        filters = ['sdss2010-u', 'sdss2010-g', 'sdss2010-r', 'sdss2010-i',
                   'sdss2010-z', 'bessell-U', 'bessell-B', 'bessell-V',
                   'bessell-R', 'bessell-I']

        for filt in filters:
            dat = self.grab_arrays(filt)
            if 'sdss' in filt:
                key = filt.split('-')
                key = key[0][:4] + '_' + key[-1]
            else:
                key = '_'.join(e for e in filt.split('-'))
            setattr(self, key, dat)

    def grab_arrays(self, bandpass):
        # Gets the wavelength array and response function arrays
        # Returns: wavelength array, respond function array, effective wavelength
        # Wavelength and effective wavelength are in units of AA
        resp = speclite.filters.load_filter(bandpass)
        wvl = jnp.linspace(resp._wavelength.min()-5, resp._wavelength.max()+5, 1000)
        return {'wvl': wvl*units.AA, 'rf': resp(wvl), 'eff_wvl':resp.effective_wavelength}
