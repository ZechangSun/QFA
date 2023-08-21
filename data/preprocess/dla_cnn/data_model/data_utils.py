""" Utilities for data """

import numpy as np
import math

from dla_cnn import defs

def pad_loglam_flux(loglam, flux, z_qso, kernel=1800, sig=None):
    # kernel = 1800    # Overriding left padding to increase it
    assert np.shape(loglam) == np.shape(flux)
    pad_loglam_upper = loglam[0] - 0.0001
    pad_loglam_lower = (math.floor(math.log10(defs.REST_RANGE[0] * (1 + z_qso)) * 10000) - kernel / 2) / 10000
    pad_loglam = np.linspace(pad_loglam_lower, pad_loglam_upper, max(0, int((pad_loglam_upper - pad_loglam_lower + 0.0001) * 10000)), dtype=np.float32)
    pad_value = np.mean(flux[0:50])
    flux_padded = np.hstack((pad_loglam*0+pad_value, flux))
    loglam_padded = np.hstack((pad_loglam, loglam))
    assert (10**loglam_padded[0])/(1+z_qso) <= defs.REST_RANGE[0]
    # Error array
    if sig is not None:
        sig_padded = np.hstack((pad_loglam*0+pad_value, sig))
        return loglam_padded, flux_padded, sig_padded
    else:
        return loglam_padded, flux_padded
