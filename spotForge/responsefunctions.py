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
            dat = self.calculate_response(filt)
            if 'sdss' in filt:
                key = filt.split('-')
                key = key[0][:4] + '_' + key[-1]
            else:
                key = '_'.join(e for e in filt.split('-'))
            setattr(self, key, dat)

    def calculate_response(self, bandpass):
        # Gets the wavelength array and response function arrays
        # Returns: wavelength array, respond function array, effective wavelength
        # Wavelength and effective wavelength are in units of AA
        resp = speclite.filters.load_filter(bandpass)
        wvl = jnp.linspace(resp._wavelength.min()-5, resp._wavelength.max()+5, 1000)
        return {'wvl': wvl*units.AA, 'rf': resp(wvl), 'eff_wvl':resp.effective_wavelength}

    def assign_response(self, filt='sdss', band='g'):
        """
        Returns the appropriate filter and band response function and wavelength
        array.

        Parameters
        -----------
        filt : str, optional
           Name of the filter. Options are `sdss` and `bessell`. Default is `sdss`.
        band : str, optional
           The bandpass letter. Options are `u, b, g, v, r, i, z`. Default is `g`.
        """
        if filt.lower() == 'sdss' and band.lower() == 'u':
            return self.sdss_u
        elif filt.lower() == 'sdss' and band.lower() == 'g':
            return self.sdss_g
        elif filt.lower() == 'sdss' and band.lower() == 'r':
            return self.sdss_r
        elif filt.lower() == 'sdss' and band.lower() == 'i':
            return self.sdss_i
        elif filt.lower() == 'sdss' and band.lower() == 'z':
            return self.sdss_z
        elif filt.lower() == 'bessell' and band.lower() == 'u':
            return self.bessell_U
        elif filt.lower() == 'bessell' and band.lower() == 'b':
            return self.bessell_B
        elif filt.lower() == 'bessell' and band.lower() == 'v':
            return self.bessell_V
        elif filt.lower() == 'bessell' and band.lower() == 'r':
            return self.bessell_R
        elif filt.lower() == 'bessell' and band.lower() == 'i':
            return self.bessell_I
        else:
            return('Bandpass not implemented.')
