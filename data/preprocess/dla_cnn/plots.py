""" Module for loading spectra, either fake or real
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import numpy as np
import pdb


import code, traceback, threading

#from dla_cnn.absorption import generate_voigt_model

# Raise warnings to errors for debugging
#import warnings
#warnings.filterwarnings('error')

# Generates a PDF visuals for a sightline and predictions
def generate_pdf(sightline, path):
    from dla_cnn.data_loader import get_lam_data
    from dla_cnn.data_loader import REST_RANGE
    from dla_cnn.absorption import generate_voigt_model

    loc_conf = sightline.prediction.loc_conf
    peaks_offset = sightline.prediction.peaks_ixs
    offset_conv_sum = sightline.prediction.offset_conv_sum
    # smoothed_sample = sightline.prediction.smoothed_loc_conf()

    PLOT_LEFT_BUFFER = 50       # The number of pixels to plot left of the predicted sightline
    dlas_counter = 0

    filename = path + "/dla-spec-%s.pdf"%sightline.id.id_string()
    pp = PdfPages(filename)

    full_lam, full_lam_rest, full_ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
    lam_rest = full_lam_rest[full_ix_dla_range]

    xlim = [REST_RANGE[0]-PLOT_LEFT_BUFFER, lam_rest[-1]]
    y = sightline.flux
    y_plot_range = np.mean(y[y > 0]) * 3.0
    ylim = [-2, y_plot_range]

    n_dlas = len(sightline.prediction.peaks_ixs)

    # Plot DLA range
    n_rows = 3 + (1 if n_dlas>0 else 0) + n_dlas
    fig = plt.figure(figsize=(20, (3.75 * (4+n_dlas)) + 0.15))
    axtxt = fig.add_subplot(n_rows,1,1)
    axsl = fig.add_subplot(n_rows,1,2)
    axloc = fig.add_subplot(n_rows,1,3)

    axsl.set_xlabel("Rest frame sightline in region of interest for DLAs with z_qso = [%0.4f]" % sightline.z_qso)
    axsl.set_ylabel("Flux")
    axsl.set_ylim(ylim)
    axsl.set_xlim(xlim)
    axsl.plot(full_lam_rest, sightline.flux, '-k')

    # Plot 0-line
    axsl.axhline(0, color='grey')

    # Plot z_qso line over sightline
    # axsl.plot((1216, 1216), (ylim[0], ylim[1]), 'k-', linewidth=2, color='grey', alpha=0.4)

    # Plot observer frame ticks
    axupper = axsl.twiny()
    axupper.set_xlim(xlim)
    xticks = np.array(axsl.get_xticks())[1:-1]
    axupper.set_xticks(xticks)
    axupper.set_xticklabels((xticks * (1 + sightline.z_qso)).astype(np.int32))

    # Sanity check
    if sightline.dlas and len(sightline.dlas) > 9:
        print("number of sightlines for {:s} is {:d}".format(
            sightline.id.id_string(), len(sightline.dlas)))

    # Plot given DLA markers over location plot
    #for dla in sightline.dlas if sightline.dlas is not None else []:
    #    dla_rest = dla.central_wavelength / (1+sightline.z_qso)
    #    axsl.plot((dla_rest, dla_rest), (ylim[0], ylim[1]), 'g--')

    # Plot localization
    axloc.set_xlabel("DLA Localization confidence & localization prediction(s)")
    axloc.set_ylabel("Identification")
    axloc.plot(lam_rest, loc_conf, color='deepskyblue')
    axloc.set_ylim([0, 1])
    axloc.set_xlim(xlim)

    # Classification results
    textresult = "Classified %s (%0.5f ra / %0.5f dec) with %d DLAs/sub dlas/Ly-B\n" \
        % (sightline.id.id_string(), sightline.id.ra, sightline.id.dec, n_dlas)

    # Plot localization histogram
    axloc.scatter(lam_rest, sightline.prediction.offset_hist, s=6, color='orange')
    axloc.plot(lam_rest, sightline.prediction.offset_conv_sum, color='green')
    axloc.plot(lam_rest, sightline.prediction.smoothed_conv_sum(), color='yellow', linestyle='-', linewidth=0.25)

    # Plot '+' peak markers
    if len(peaks_offset) > 0:
        axloc.plot(lam_rest[peaks_offset], np.minimum(1, offset_conv_sum[peaks_offset]), '+', mew=5, ms=10, color='green', alpha=1)

    #
    # For loop over each DLA identified
    #
    for dlaix, peak in zip(range(0,n_dlas), peaks_offset):
        # Some calculations that will be used multiple times
        dla_z = lam_rest[peak] * (1 + sightline.z_qso) / 1215.67 - 1

        # Sightline plot transparent marker boxes
        axsl.fill_between(lam_rest[peak - 10:peak + 10], y_plot_range, -2, color='green', lw=0, alpha=0.1)
        axsl.fill_between(lam_rest[peak - 30:peak + 30], y_plot_range, -2, color='green', lw=0, alpha=0.1)
        axsl.fill_between(lam_rest[peak - 50:peak + 50], y_plot_range, -2, color='green', lw=0, alpha=0.1)
        axsl.fill_between(lam_rest[peak - 70:peak + 70], y_plot_range, -2, color='green', lw=0, alpha=0.1)

        # Plot column density measures with bar plots
        # density_pred_per_this_dla = sightline.prediction.density_data[peak-40:peak+40]
        dlas_counter += 1
        # mean_col_density_prediction = float(np.mean(density_pred_per_this_dla))
        density_pred_per_this_dla, mean_col_density_prediction, std_col_density_prediction, bias_correction = \
            sightline.prediction.get_coldensity_for_peak(peak)

        pltix = fig.add_subplot(n_rows, 1, 5+dlaix)
        pltix.bar(np.arange(0, density_pred_per_this_dla.shape[0]), density_pred_per_this_dla, 0.25)
        pltix.set_xlabel("Individual Column Density estimates for peak @ %0.0fA, +/- 0.3 of mean. Bias adjustment of %0.3f added. " %
                             (lam_rest[peak], float(bias_correction)) +
                             "Mean: %0.3f - Median: %0.3f - Stddev: %0.3f" %
                             (mean_col_density_prediction, float(np.median(density_pred_per_this_dla)),
                              float(std_col_density_prediction)))
        pltix.set_ylim([mean_col_density_prediction - 0.3, mean_col_density_prediction + 0.3])
        pltix.plot(np.arange(0, density_pred_per_this_dla.shape[0]),
                       np.ones((density_pred_per_this_dla.shape[0]), np.float32) * mean_col_density_prediction)
        pltix.set_ylabel("Column Density")

        # Add DLA to test result
        absorber_type = "Ly-b" if sightline.is_lyb(peak) else "DLA" if mean_col_density_prediction >= 20.3 else "sub dla"
        dla_text = \
            "%s at: %0.0fA rest / %0.0fA observed / %0.4f z, w/ confidence %0.2f, has Column Density: %0.3f" \
            % (absorber_type,
               lam_rest[peak],
               lam_rest[peak] * (1 + sightline.z_qso),
               dla_z,
               min(1.0, float(sightline.prediction.offset_conv_sum[peak])),
               mean_col_density_prediction)
        textresult += " > " + dla_text + "\n"

        #
        # Plot DLA zoom view with voigt overlay
        #

        dla_min_text = \
            "%0.0fA rest / %0.0fA observed - NHI %0.3f" \
            % (lam_rest[peak],
               lam_rest[peak] * (1 + sightline.z_qso),
               mean_col_density_prediction)

        # Generate the Voigt profile
        absorber = dict(z_dla=dla_z, column_density=mean_col_density_prediction)
        voigt_wave, voigt_model, ixs_mypeaks = generate_voigt_model(sightline, absorber)
        inax = fig.add_subplot(n_rows, n_dlas, n_dlas*3+dlaix+1)
        inax.plot(full_lam, sightline.flux, '-k', lw=1.2)
        inax.plot(full_lam[ixs_mypeaks], sightline.flux[ixs_mypeaks], '+', mew=5, ms=10, color='green', alpha=1)
        inax.plot(voigt_wave, voigt_model, 'g--', lw=3.0)
        inax.set_ylim(ylim)
        # convert peak to index into full_lam range for plotting
        peak_full_lam = np.nonzero(np.cumsum(full_ix_dla_range) > peak)[0][0]
        inax.set_xlim([full_lam[peak_full_lam-150],full_lam[peak_full_lam+150]])
        inax.axhline(0, color='grey')

        #
        # Plot legend on location graph
        #
        axloc.legend(['DLA classifier', 'Localization', 'DLA peak', 'Localization histogram'],
                         bbox_to_anchor=(1.0, 1.05))


    # Display text
    axtxt.text(0, 0, textresult, family='monospace', fontsize='xx-large')
    axtxt.get_xaxis().set_visible(False)
    axtxt.get_yaxis().set_visible(False)
    axtxt.set_frame_on(False)

    fig.tight_layout()
    pp.savefig(figure=fig)
    pp.close()
    plt.close('all')
    print("Wrote figure to {:s}".format(filename))

