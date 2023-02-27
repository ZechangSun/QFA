""" Module to vette results against Human catalogs
     SDSS-DR5 (JXP) and BOSS (Notredaeme)
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import pdb


import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u

from linetools import utils as ltu
from pyigm.surveys.llssurvey import LLSSurvey
from pyigm.surveys.dlasurvey import DLASurvey


def json_to_sdss_dlasurvey(json_file, sdss_survey, add_pf=True, debug=False):
    """ Convert JSON output file to a DLASurvey object
    Assumes SDSS bookkeeping for sightlines (i.e. PLATE, FIBER)

    Parameters
    ----------
    json_file : str
      Full path to the JSON results file
    sdss_survey : DLASurvey
      SDSS survey, usually human (e.g. JXP for DR5)
    add_pf : bool, optional
      Add plate/fiber to DLAs in sdss_survey

    Returns
    -------
    ml_survey : LLSSurvey
      Survey object for the LLS

    """
    print("Loading SDSS Survey from JSON file {:s}".format(json_file))
    # imports
    from pyigm.abssys.dla import DLASystem
    from pyigm.abssys.lls import LLSSystem
    # Fiber key
    for fkey in ['FIBER', 'FIBER_ID', 'FIB']:
        if fkey in sdss_survey.sightlines.keys():
            break
    # Read
    ml_results = ltu.loadjson(json_file)
    use_platef = False
    if 'plate' in ml_results[0].keys():
        use_platef = True
    else:
        if 'id' in ml_results[0].keys():
            use_id = True
    # Init
    #idict = dict(plate=[], fiber=[], classification_confidence=[],  # FOR v2
    #             classification=[], ra=[], dec=[])
    idict = dict(ra=[], dec=[])
    if use_platef:
        for key in ['plate', 'fiber', 'mjd']:
            idict[key] = []
    ml_tbl = Table()
    ml_survey = LLSSurvey()
    systems = []
    in_ml = np.array([False]*len(sdss_survey.sightlines))
    # Loop
    for obj in ml_results:
        # Sightline
        for key in idict.keys():
            idict[key].append(obj[key])
        # DLAs
        #if debug:
        #    if (obj['plate'] == 1366) & (obj['fiber'] == 614):
        #        sv_coord = SkyCoord(ra=obj['ra'], dec=obj['dec'], unit='deg')
        #        print("GOT A MATCH IN RESULTS FILE")
        for idla in obj['dlas']:
            """
            dla = DLASystem((sdss_survey.sightlines['RA'][mt[0]],
                             sdss_survey.sightlines['DEC'][mt[0]]),
                            idla['spectrum']/(1215.6701)-1., None,
                            idla['column_density'])
            """
            if idla['z_dla'] < 1.8:
                continue
            isys = LLSSystem((obj['ra'],obj['dec']),
                    idla['z_dla'], None, NHI=idla['column_density'], zem=obj['z_qso'])
            isys.confidence = idla['dla_confidence']
            if use_platef:
                isys.plate = obj['plate']
                isys.fiber = obj['fiber']
            elif use_id:
                plate, fiber = [int(spl) for spl in obj['id'].split('-')]
                isys.plate = plate
                isys.fiber = fiber
            # Save
            systems.append(isys)
    # Connect to sightlines
    ml_coord = SkyCoord(ra=idict['ra'], dec=idict['dec'], unit='deg')
    s_coord = SkyCoord(ra=sdss_survey.sightlines['RA'], dec=sdss_survey.sightlines['DEC'], unit='deg')
    idx, d2d, d3d = match_coordinates_sky(s_coord, ml_coord, nthneighbor=1)
    used = d2d < 1.*u.arcsec
    for iidx in np.where(~used)[0]:
        print("Sightline RA={:g}, DEC={:g} was not used".format(sdss_survey.sightlines['RA'][iidx],
                                                                sdss_survey.sightlines['DEC'][iidx]))
    # Add plate/fiber to statistical DLAs
    if add_pf:
        dla_coord = sdss_survey.coord
        idx2, d2d, d3d = match_coordinates_sky(dla_coord, s_coord, nthneighbor=1)
        if np.min(d2d.to('arcsec').value) > 1.:
            raise ValueError("Bad match to sightlines")
        for jj,igd in enumerate(np.where(sdss_survey.mask)[0]):
            dla = sdss_survey._abs_sys[igd]
            try:
                dla.plate = sdss_survey.sightlines['PLATE'][idx2[jj]]
            except IndexError:
                pdb.set_trace()
            dla.fiber = sdss_survey.sightlines[fkey][idx2[jj]]
    # Finish
    ml_survey._abs_sys = systems
    if debug:
        ml2_coord = ml_survey.coord
        minsep = np.min(sv_coord.separation(ml2_coord))
        minsep2 = np.min(sv_coord.separation(s_coord))
        tmp = sdss_survey.sightlines[used]
        t_coord = SkyCoord(ra=tmp['RA'], dec=tmp['DEC'], unit='deg')
        minsep3 = np.min(sv_coord.separation(t_coord))
        pdb.set_trace()
    ml_survey.sightlines = sdss_survey.sightlines[used]
    for key in idict.keys():
        ml_tbl[key] = idict[key]
    ml_survey.ml_tbl = ml_tbl
    # Return
    return ml_survey


def vette_dlasurvey(ml_survey, sdss_survey, fig_root='tmp', lyb_cut=True,
                    dz_toler=0.03, debug=False):
    """
    Parameters
    ----------
    ml_survey : IGMSurvey
      Survey describing the Machine Learning results
    sdss_survey : DLASurvey
      SDSS survey, usually human (e.g. JXP for DR5)
    fig_root : str, optional
      Root string for figures generated
    lyb_cut : bool, optional
      Cut surveys at Lyb in QSO rest-frame.
      Recommended until LLS, Lyb and OVI is dealt with
    dz_toler : float, optional
      Tolerance for matching in redshift

    Returns
    -------
    false_neg : list
      List of systems that are false negatives from SDSS -> ML
    midx : list
      List of indices matching SDSS -> ML
    """
    from pyigm.surveys import dlasurvey as pyis_ds
    reload(pyis_ds)
    # Cut at Lyb
    if lyb_cut:
        for survey in [ml_survey, sdss_survey]:
            # Alter Z_START
            zlyb = (1+survey.sightlines['ZEM']).data*1026./1215.6701 - 1.
            survey.sightlines['Z_START'] = np.maximum(survey.sightlines['Z_START'], zlyb)
            # Mask
            mask = pyis_ds.dla_stat(survey, survey.sightlines, zem_tol=0.2)  # Errors in zem!
            survey.mask = mask
        print("Done cutting on Lyb")

    # Setup coords
    ml_coords = ml_survey.coord
    ml_z = ml_survey.zabs
    s_coords = sdss_survey.coord
    s_z = sdss_survey.zabs
#    if debug:
#        miss_coord = SkyCoord(ra=174.35545833333333,dec=44.585,unit='deg')
#        minsep = np.min(miss_coord.separation(ml_coords))
#        s_coord = SkyCoord(ra=ml_survey.sightlines['RA'], dec=ml_survey.sightlines['DEC'], unit='deg')
#        isl = np.argmin(miss_coord.separation(s_coord))

    # Match from SDSS and record false negatives
    false_neg = []
    midx = []
    for igd in np.where(sdss_survey.mask)[0]:
        isys = sdss_survey._abs_sys[igd]
        # Match?
        gd_radec = np.where(isys.coord.separation(ml_coords) < 1*u.arcsec)[0]
        sep = isys.coord.separation(ml_coords)
        if len(gd_radec) == 0:
            false_neg.append(isys)
            midx.append(-1)
        else:
            gdz = np.abs(ml_z[gd_radec] - isys.zabs) < dz_toler
            # Only require one match
            if np.sum(gdz) > 0:
                iz = np.argmin(np.abs(ml_z[gd_radec] - isys.zabs))
                midx.append(gd_radec[iz])
            else:
                false_neg.append(isys)
                midx.append(-1)
        if debug:
            if (isys.plate == 1366) & (isys.fiber == 614):
                pdb.set_trace()

    # Match from ML and record false positives
    false_pos = []
    pidx = []
    for igd in np.where(ml_survey.mask)[0]:
        isys = ml_survey._abs_sys[igd]
        # Match?
        gd_radec = np.where(isys.coord.separation(s_coords) < 1*u.arcsec)[0]
        sep = isys.coord.separation(s_coords)
        if len(gd_radec) == 0:
            false_pos.append(isys)
            pidx.append(-1)
        else:
            gdz = np.abs(s_z[gd_radec] - isys.zabs) < dz_toler
            # Only require one match
            if np.sum(gdz) > 0:
                iz = np.argmin(np.abs(s_z[gd_radec] - isys.zabs))
                pidx.append(gd_radec[iz])
            else:
                false_pos.append(isys)
                pidx.append(-1)

    # Return
    return false_neg, np.array(midx), false_pos

def mk_false_neg_table(false_neg, outfil):
    """ Generate a simple CSV file of false negatives

    Parameters
    ----------
    false_neg : list
      List of false negative systems
    outfil : str

    Returns
    -------

    """
    # Parse
    ra, dec = [], []
    zabs, zem = [], []
    NHI = []
    plate, fiber = [], []
    for ifneg in false_neg:
        ra.append(ifneg.coord.ra.value)
        dec.append(ifneg.coord.dec.value)
        zabs.append(ifneg.zabs)
        zem.append(ifneg.zem)
        NHI.append(ifneg.NHI)
        plate.append(ifneg.plate)
        fiber.append(ifneg.fiber)
    # Generate a Table
    fneg_tbl = Table()
    fneg_tbl['RA'] = ra
    fneg_tbl['DEC'] = dec
    fneg_tbl['zabs'] = zabs
    fneg_tbl['zem'] = zem
    fneg_tbl['NHI'] = NHI
    fneg_tbl['plate'] = plate
    fneg_tbl['fiber'] = fiber
    # Write
    print("Writing false negative file: {:s}".format(outfil))
    fneg_tbl.write(outfil, format='ascii.csv')#, overwrite=True)


def fig_dzdnhi(ml_survey, sdss_survey, midx, outfil='fig_dzdnhi.pdf'):
    """  Compare zabs and NHI between SDSS and ML

    Parameters
    ----------
    ml_survey : IGMSurvey
      Survey describing the Machine Learning results
      This should be masked according to the vetting
    sdss_survey : DLASurvey
      SDSS survey, usually human (e.g. JXP for DR5)
      This should be masked according to the vetting
    midx : list
      List of indices matching SDSS -> ML
    outfil : str, optional
      Input None to plot to screen

    Returns
    -------

    """
    # z, NHI
    z_sdss = sdss_survey.zabs
    z_ml = ml_survey.zabs
    NHI_sdss = sdss_survey.NHI
    NHI_ml = ml_survey.NHI
    # deltas
    dz = []
    dNHI = []
    for qq,idx in enumerate(midx):
        if idx < 0:
            continue
        # Match
        dz.append(z_sdss[qq]-z_ml[idx])
        dNHI.append(NHI_sdss[qq]-NHI_ml[idx])

    # Figure
    if outfil is not None:
        pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8, 5))
    plt.clf()
    gs = gridspec.GridSpec(1, 2)
    # dz
    ax = plt.subplot(gs[0])
    ax.hist(dz, color='green', bins=20)#, normed=True)#, bins=20 , zorder=1)
    #ax.text(0.05, 0.74, lbl3, transform=ax.transAxes, color=wcolor, size=csz, ha='left')
    ax.set_xlim(-0.03, 0.03)
    ax.set_xlabel(r'$\delta z$ [SDSS-ML]')
    # NHI
    ax = plt.subplot(gs[1])
    ax.hist(dNHI, color='blue', bins=20)#, normed=True)#, bins=20 , zorder=1)
    #ax.text(0.05, 0.74, lbl3, transform=ax.transAxes, color=wcolor, size=csz, ha='left')
    #ax.set_xlim(-0.03, 0.03)
    ax.set_xlabel(r'$\Delta \log N_{\rm HI}$ [SDSS-ML]')
    #
    # End
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    if outfil is not None:
        print('Writing {:s}'.format(outfil))
        pp.savefig()
        pp.close()
        plt.close()
    else:
        plt.show()


def fig_falseneg(ml_survey, sdss_survey, false_neg, outfil='fig_falseneg.pdf'):
    """   Figure on false negatives

    Parameters
    ----------
    ml_survey : IGMSurvey
      Survey describing the Machine Learning results
      This should be masked according to the vetting
    sdss_survey : DLASurvey
      SDSS survey, usually human (e.g. JXP for DR5)
      This should be masked according to the vetting
    midx : list
      List of indices matching SDSS -> ML
    false_neg : list
      List of false negatives
    outfil : str, optional
      Input None to plot to screen

    Returns
    -------

    """
    # Generate some lists
    zabs_false = [isys.zabs for isys in false_neg]
    zem_false = [isys.zem for isys in false_neg]
    NHI_false = [isys.NHI for isys in false_neg]

    # Figure
    if outfil is not None:
        pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8, 5))
    plt.clf()
    gs = gridspec.GridSpec(2, 2)
    # zabs
    ax = plt.subplot(gs[0])
    ax.hist(zabs_false, color='green', bins=20)#, normed=True)#, bins=20 , zorder=1)
    ax.set_xlabel(r'$z_{\rm abs}$')
    # zem
    ax = plt.subplot(gs[1])
    ax.hist(zem_false, color='red', bins=20)#, normed=True)#, bins=20 , zorder=1)
    ax.set_xlabel(r'$z_{\rm qso}$')
    # NHI
    ax = plt.subplot(gs[2])
    ax.hist(NHI_false, color='blue', bins=20)#, normed=True)#, bins=20 , zorder=1)
    ax.set_xlabel(r'$\log \, N_{\rm HI}$')
    # End
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    if outfil is not None:
        print('Writing {:s}'.format(outfil))
        pp.savefig()
        pp.close()
        plt.close()
    else:
        plt.show()


def dr5_for_david():
    """ Generate a Table for David
    """
    # imports
    from pyigm.abssys.dla import DLASystem
    from pyigm.abssys.lls import LLSSystem
    sdss_survey = DLASurvey.load_SDSS_DR5()
    # Fiber key
    for fkey in ['FIBER', 'FIBER_ID', 'FIB']:
        if fkey in sdss_survey.sightlines.keys():
            break
    # Init
    #idict = dict(plate=[], fiber=[], classification_confidence=[],  # FOR v2
    #             classification=[], ra=[], dec=[])
    # Connect to sightlines
    s_coord = SkyCoord(ra=sdss_survey.sightlines['RA'], dec=sdss_survey.sightlines['DEC'], unit='deg')
    # Add plate/fiber to statistical DLAs
    dla_coord = sdss_survey.coord
    idx2, d2d, d3d = match_coordinates_sky(dla_coord, s_coord, nthneighbor=1)
    if np.min(d2d.to('arcsec').value) > 1.:
        raise ValueError("Bad match to sightlines")
    plates, fibers = [], []
    for jj,igd in enumerate(np.where(sdss_survey.mask)[0]):
        dla = sdss_survey._abs_sys[igd]
        try:
            dla.plate = sdss_survey.sightlines['PLATE'][idx2[jj]]
        except IndexError:
            pdb.set_trace()
        dla.fiber = sdss_survey.sightlines[fkey][idx2[jj]]
        plates.append(sdss_survey.sightlines['PLATE'][idx2[jj]])
        fibers.append(sdss_survey.sightlines[fkey][idx2[jj]])
    # Write
    dtbl = Table()
    dtbl['plate'] = plates
    dtbl['fiber'] = fibers
    dtbl['zabs'] = sdss_survey.zabs
    dtbl['NHI'] = sdss_survey.NHI
    dtbl.write('results/dr5_for_david.ascii', format='ascii')
    # Write sightline info
    stbl = sdss_survey.sightlines[['PLATE', 'FIB', 'Z_START', 'Z_END', 'RA', 'DEC']]
    gdsl = stbl['Z_END'] > stbl['Z_START']
    stbl[gdsl].write('results/dr5_sightlines_for_david.ascii', format='ascii')

def main(flg_tst, sdss=None, ml_survey=None):

    # Load JSON for DR5
    if (flg_tst % 2**1) >= 2**0:
        if sdss is None:
            sdss = DLASurvey.load_SDSS_DR5()
        #ml_survey = json_to_sdss_dlasurvey('../results/dr5_v1_predictions.json', sdss)
        ml_survey = json_to_sdss_dlasurvey('../results/dr5_v2_results.json', sdss)

    # Vette
    if (flg_tst % 2**2) >= 2**1:
        if ml_survey is None:
            sdss = DLASurvey.load_SDSS_DR5()
            ml_survey = json_to_sdss_dlasurvey('../results/dr5_v2_results.json', sdss)
        vette_dlasurvey(ml_survey, sdss)

    # Vette v5 and generate CSV
    if (flg_tst % 2**3) >= 2**2:
        if ml_survey is None:
            sdss = DLASurvey.load_SDSS_DR5()
            ml_survey = json_to_sdss_dlasurvey('../results/dr5_v5_predictions.json', sdss)
        false_neg, midx, _ = vette_dlasurvey(ml_survey, sdss)
        # CSV of false negatives
        mk_false_neg_table(false_neg, '../results/false_negative_DR5_v5.csv')

    # Vette v6 and generate CSV
    if (flg_tst % 2**4) >= 2**3:
        if ml_survey is None:
            sdss = DLASurvey.load_SDSS_DR5()
            ml_survey = json_to_sdss_dlasurvey('../results/dr5_v6.1_results.json', sdss)
        false_neg, midx, _ = vette_dlasurvey(ml_survey, sdss)
        # CSV of false negatives
        mk_false_neg_table(false_neg, '../results/false_negative_DR5_v6.1.csv')

    # Vette gensample v2
    if (flg_tst % 2**5) >= 2**4:
        if ml_survey is None:
            sdss = DLASurvey.load_SDSS_DR5()
            ml_survey = json_to_sdss_dlasurvey('../results/results_catalog_dr7_model_gensample_v2.json',sdss)
        false_neg, midx, false_pos = vette_dlasurvey(ml_survey, sdss)
        # CSV of false negatives
        mk_false_neg_table(false_neg, '../results/false_negative_DR5_v2_gen.csv')
        mk_false_neg_table(false_pos, '../results/false_positives_DR5_v2_gen.csv')

    # Vette gensample v4.3.1
    if flg_tst & (2**5):
        if ml_survey is None:
            sdss = DLASurvey.load_SDSS_DR5()
            ml_survey = json_to_sdss_dlasurvey('../results/results_model_4.3.1_data_dr5.json',sdss)
        false_neg, midx, false_pos = vette_dlasurvey(ml_survey, sdss)
        # CSV of false negatives
        mk_false_neg_table(false_neg, '../results/false_negative_DR5_v4.3.1_gen.csv')
        mk_false_neg_table(false_pos, '../results/false_positives_DR5_v4.3.1_gen.csv')

    if flg_tst & (2**6):
        dr5_for_david()

# Test
if __name__ == '__main__':
    flg_tst = 0
    #flg_tst += 2**0   # Load JSON for DR5
    #flg_tst += 2**1   # Vette
    #flg_tst += 2**2   # v5
    #flg_tst += 2**3   # v6.1
    #flg_tst += 2**4   # v2 of gensample
    #flg_tst += 2**5   # v4.3.1 of gensample
    flg_tst += 2**6   # Generate DR5 table for David

    main(flg_tst)
