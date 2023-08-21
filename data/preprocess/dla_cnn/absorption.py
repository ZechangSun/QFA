""" Module for loading spectra, either fake or real
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import os, sys
import numpy as np
import pdb

from dla_cnn.data_loader import REST_RANGE


# Raise warnings to errors for debugging
import warnings
#warnings.filterwarnings('error')

def add_abs_to_sightline(sightline):
    from dla_cnn.data_loader import get_lam_data
    #if REST_RANGE is None:
    #    from dla_cnn.data_loader import REST_RANGE
    #
    dlas = []
    subdlas = []
    lybs = []

    # Loop through peaks which identify a DLA
    # (peaks, peaks_uncentered, smoothed_sample, ixs_left, ixs_right, offset_hist, offset_conv_sum, peaks_offset) \
    #     = peaks_data[ix]
    lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
    for peak in sightline.prediction.peaks_ixs:
        peak_lam_rest = lam_rest[ix_dla_range][peak]
        peak_lam_spectrum = peak_lam_rest * (1 + sightline.z_qso)

        # mean_col_density_prediction = np.mean(density_data[peak-40:peak+40])
        # std_col_density_prediction = np.std(density_data[peak-40:peak+40])
        z_dla = float(peak_lam_spectrum) / 1215.67 - 1
        _, mean_col_density_prediction, std_col_density_prediction, bias_correction = \
            sightline.prediction.get_coldensity_for_peak(peak)

        absorber_type = "LYB" if sightline.is_lyb(peak) else "DLA" if mean_col_density_prediction >= 20.3 else "SUBDLA"
        dla_sub_lyb = lybs if absorber_type == "LYB" else dlas if absorber_type == "DLA" else subdlas

        # Should add S/N at peak
        abs_dict =  {
            'rest': float(peak_lam_rest),
            'spectrum': float(peak_lam_spectrum),
            'z_dla':float(z_dla),
            'dla_confidence': min(1.0,float(sightline.prediction.offset_conv_sum[peak])),
            'column_density': float(mean_col_density_prediction),
            'std_column_density': float(std_col_density_prediction),
            'column_density_bias_adjust': float(bias_correction),
            'type': absorber_type
        }
        #get_s2n_for_absorbers(sightline, lam, [abs_dict])  # SLOWED CODE DOWN TOO MUCH
        dla_sub_lyb.append(abs_dict)
    # Save
    sightline.dlas = dlas
    sightline.subdlas = subdlas
    sightline.lybs = lybs

def generate_voigt_model(sightline, absorber):
    """ Generate a continuum-scaled Voigt profile for the absorber
    :param sightline:
    :param absorber:
    :param REST_RANGE:
    :return:
    voigt_wave, voigt_model
    """
    from dla_cnn.data_loader import get_lam_data
    #from dla_cnn.data_loader import generate_voigt_profile
    #from dla_cnn.data_loader import get_peaks_for_voigt_scaling
    from scipy.stats import chisquare
    from scipy.optimize import minimize
    #if REST_RANGE is None:
    #    from dla_cnn.data_loader import REST_RANGE
    # Wavelengths
    full_lam, full_lam_rest, full_ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
    # Generate the voigt model using astropy, linetools, etc.
    voigt_flux, voigt_wave = generate_voigt_profile(absorber['z_dla'],
                                                    absorber['column_density'], full_lam)
    # get peaks
    ixs_mypeaks = get_peaks_for_voigt_scaling(sightline, voigt_flux)
    # get indexes where voigt profile is between 0.2 and 0.95
    observed_values = sightline.flux[ixs_mypeaks]
    expected_values = voigt_flux[ixs_mypeaks]
    # Minimize scale variable using chi square measure
    opt = minimize(lambda scale: chisquare(observed_values, expected_values * scale)[0], 1)
    opt_scale = opt.x[0]
    #from IPython import embed; embed()

    # Return
    return voigt_wave, voigt_flux*opt_scale, ixs_mypeaks

def generate_voigt_profile(dla_z, mean_col_density_prediction, full_lam):
    from linetools.spectralline import AbsLine
    from linetools.analysis.voigt import voigt_from_abslines
    from astropy import units as u
    with open(os.devnull, 'w') as devnull:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Hack to avoid AbsLine spamming us with print statements
            stdout = sys.stdout
            sys.stdout = devnull

            abslin = AbsLine(1215.670 * 0.1 * u.nm, z=dla_z)
            abslin.attrib['N'] = 10 ** mean_col_density_prediction / u.cm ** 2  # log N
            abslin.attrib['b'] = 25. * u.km / u.s  # b
            # print dla_z, mean_col_density_prediction, full_lam, full_lam.shape
            # try:
            #vmodel = voigt_from_abslines(full_lam.astype(np.float16) * u.AA, abslin, fwhm=3, debug=False)
            vmodel = voigt_from_abslines(full_lam * u.AA, abslin, fwhm=3, debug=False)
            # except TypeError as e:
            #     import pdb; pdb.set_trace()
            voigt_flux = vmodel.data['flux'].data[0]
            voigt_wave = vmodel.data['wave'].data[0]
            # clear some bad values at beginning / end of voigt_flux
            voigt_flux[0:10] = 1
            voigt_flux[-10:len(voigt_flux) + 1] = 1

            sys.stdout = stdout

    return voigt_flux, voigt_wave

# Returns peaks used for voigt scaling, removes outlier and ensures enough points for good scaling
def get_peaks_for_voigt_scaling(sightline, voigt_flux):
    from scipy.signal import find_peaks_cwt
    iteration_count = 0
    ixs_mypeaks_outliers_removed = []

    # Loop to try different find_peak values if we don't get enough peaks with one try
    while iteration_count < 10 and len(ixs_mypeaks_outliers_removed) < 5:
        peaks = np.array(find_peaks_cwt(sightline.flux, np.arange(1, 2+iteration_count)))
        ixs = np.where((voigt_flux > 0.2) & (voigt_flux < 0.95))
        ixs_mypeaks = np.intersect1d(ixs, peaks)

        # Remove any points > 1.5 standard deviations from the mean (poor mans outlier removal)
        peaks_mean = np.mean(sightline.flux[ixs_mypeaks]) if len(ixs_mypeaks)>0 else 0
        peaks_std = np.std(sightline.flux[ixs_mypeaks]) if len(ixs_mypeaks)>0 else 0

        ixs_mypeaks_outliers_removed = ixs_mypeaks[np.abs(sightline.flux[ixs_mypeaks] - peaks_mean) < (peaks_std * 1.5)]
        iteration_count += 1


    return ixs_mypeaks_outliers_removed


# Estimate S/N at an absorber
def get_s2n_for_absorbers(sightline, lam, absorbers, nsamp=20):
    from scipy.stats import chisquare
    from scipy.optimize import minimize
    if len(absorbers) == 0:
        return
    # Loop on the DLAs
    for jj in range(len(absorbers)):
        # Find the peak
        isys = absorbers[jj]
        # Get the Voigt (to avoid it)
        voigt_flux, voigt_wave = generate_voigt_profile(isys['z_dla'], isys['column_density'], lam)
        # get peaks
        ixs_mypeaks = get_peaks_for_voigt_scaling(sightline, voigt_flux)
        if len(ixs_mypeaks) < 2:
            s2n = 1.  # KLUDGE
        else:
            # get indexes where voigt profile is between 0.2 and 0.95
            observed_values = sightline.flux[ixs_mypeaks]
            expected_values = voigt_flux[ixs_mypeaks]
            # Minimize scale variable using chi square measure for signal
            opt = minimize(lambda scale: chisquare(observed_values, expected_values * scale)[0], 1)
            opt_scale = opt.x[0]
            # Noise
            core = voigt_flux < 0.8
            rough_noise = np.median(sightline.sig[core])
            if rough_noise == 0:  # Occasional bad data in error array
                s2n = 0.1
            else:
                s2n = opt_scale/rough_noise
        isys['s2n'] = s2n
        '''  Another algorithm
        # Core
        core = np.where(voigt_flux < 0.8)[0]
        # Fluxes -- Take +/-nsamp away from core
        flux_for_stats = np.concatenate([sightline.flux[core[0]-nsamp:core[0]], sightline.flux[core[1]:core[1]+nsamp]])
        # Sort
        asrt = np.argsort(flux_for_stats)
        rough_signal = flux_for_stats[asrt][int(0.9*len(flux_for_stats))]
        rough_noise = np.median(sightline.sig[core])
        #
        s2n = rough_signal/rough_noise
        '''
    return


def voigt_from_sightline(sightline, inp):
    """ Wrapper to generate the Voigt for a given sightline and

    Parameters
    ----------
    sightline
    inp : int, float (coming)

    Returns
    -------
    voigt_wave
    voigt_model
    ixs_mypeaks
    """
    from dla_cnn.data_loader import get_lam_data
    # Setup
    peaks_offset = sightline.prediction.peaks_ixs
    full_lam, full_lam_rest, full_ix_dla_range = get_lam_data(sightline.loglam,
                                                              sightline.z_qso, REST_RANGE)
    # Input
    if isinstance(inp,int):
        peakid = inp
    else:
        raise IOError("Only taking peakid so far")
    # Grab peak
    peak = peaks_offset[peakid]
    # z, NHI
    lam_rest = full_lam_rest[full_ix_dla_range]
    dla_z = lam_rest[peak] * (1 + sightline.z_qso) / 1215.67 - 1
    density_pred_per_this_dla, mean_col_density_prediction, std_col_density_prediction, bias_correction = \
        sightline.prediction.get_coldensity_for_peak(peak)
    # Absorber and voigt
    absorber = dict(z_dla=dla_z, column_density=mean_col_density_prediction)
    return generate_voigt_model(sightline, absorber)
