""" Module for loading files and results from CNN
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import os, sys
import numpy as np

from pkg_resources import resource_filename

from scipy.signal import find_peaks_cwt

from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord

from linetools import utils as ltu

from pyigm.abssys.dla import DLASystem
from pyigm.abssys.lls import LLSSystem
from pyigm.surveys.llssurvey import LLSSurvey
from pyigm.surveys.dlasurvey import DLASurvey

dr7_file = resource_filename('dla_cnn', 'catalogs/sdss_dr7/predictions_SDSSDR7.json')


def load_ml_file(pred_file):
    """ Load the search results from the CNN into a DLASurvey object
    Parameters
    ----------
    pred_file

    Returns
    -------
    ml_llssurvey: LLSSurvey
    ml_dlasusrvey: DLASurvey
    """
    print("Loading {:s}.  Please be patient..".format(pred_file))
    # Read
    ml_results = ltu.loadjson(pred_file)
    use_platef = False
    if 'plate' in ml_results[0].keys():
        use_platef = True
    else:
        if 'id' in ml_results[0].keys():
            use_id = True
    # Init
    idict = dict(ra=[], dec=[], plate=[], fiber=[])
    if use_platef:
        for key in ['plate', 'fiber', 'mjd']:
            idict[key] = []
    dlasystems = []
    llssystems = []

    # Generate coords to speed things up
    for obj in ml_results:
        for key in ['ra', 'dec']:
            idict[key].append(obj[key])
    ml_coords = SkyCoord(ra=idict['ra'], dec=idict['dec'], unit='deg')
    ra_names = ml_coords.icrs.ra.to_string(unit=u.hour,sep='',pad=True)
    dec_names = ml_coords.icrs.dec.to_string(sep='',pad=True,alwayssign=True)
    vlim = [-500., 500.]*u.km/u.s
    dcoord = SkyCoord(ra=0., dec=0., unit='deg')

    # Loop on list
    didx, lidx = [], []
    print("Looping on sightlines..")
    for tt,obj in enumerate(ml_results):
        #if (tt % 100) == 0:
        #    print('tt: {:d}'.format(tt))
        # Sightline
        if use_id:
            plate, fiber = [int(spl) for spl in obj['id'].split('-')]
            idict['plate'].append(plate)
            idict['fiber'].append(fiber)

        # Systems
        for ss,syskey in enumerate(['dlas', 'subdlas']):
            for idla in obj[syskey]:
                name = 'J{:s}{:s}_z{:.3f}'.format(ra_names[tt], dec_names[tt], idla['z_dla'])
                if ss == 0:
                    isys = DLASystem(dcoord, idla['z_dla'], vlim, NHI=idla['column_density'], zem=obj['z_qso'], name=name)
                else:
                    isys = LLSSystem(dcoord, idla['z_dla'], vlim, NHI=idla['column_density'], zem=obj['z_qso'], name=name)
                isys.confidence = idla['dla_confidence']
                isys.s2n = idla['s2n']
                if use_platef:
                    isys.plate = obj['plate']
                    isys.fiber = obj['fiber']
                elif use_id:
                    isys.plate = plate
                    isys.fiber = fiber
                # Save
                if ss == 0:
                    didx.append(tt)
                    dlasystems.append(isys)
                else:
                    lidx.append(tt)
                    llssystems.append(isys)
    # Generate sightline tables
    sightlines = Table()
    sightlines['RA'] = idict['ra']
    sightlines['DEC'] = idict['dec']
    sightlines['PLATE'] = idict['plate']
    sightlines['FIBERID'] = idict['fiber']
    # Surveys
    ml_llssurvey = LLSSurvey()
    ml_llssurvey.sightlines = sightlines.copy()
    ml_llssurvey._abs_sys = llssystems
    ml_llssurvey.coords = ml_coords[np.array(lidx)]

    ml_dlasurvey = DLASurvey()
    ml_dlasurvey.sightlines = sightlines.copy()
    ml_dlasurvey._abs_sys = dlasystems
    ml_dlasurvey.coords = ml_coords[np.array(didx)]

    # Return
    return ml_llssurvey, ml_dlasurvey

def load_ml_dr7():
    """ Load the search results from the SDSS DR7 dataset into
    a DLASurvey object
    """
    print("Loading SDSSDR7 model.  Please be patient..")
    # Read
    ml_results = ltu.loadjson(dr7_file)
    use_platef = False
    if 'plate' in ml_results[0].keys():
        use_platef = True
    else:
        if 'id' in ml_results[0].keys():
            use_id = True
    # Init
    #idict = dict(plate=[], fiber=[], classification_confidence=[],  # FOR v2
    #             classification=[], ra=[], dec=[])
    idict = dict(ra=[], dec=[], plate=[], fiber=[])
    if use_platef:
        for key in ['plate', 'fiber', 'mjd']:
            idict[key] = []
    dlasystems = []
    llssystems = []

    # Generate coords to speed things up
    for obj in ml_results:
        for key in ['ra', 'dec']:
            idict[key].append(obj[key])
    ml_coords = SkyCoord(ra=idict['ra'], dec=idict['dec'], unit='deg')
    ra_names = ml_coords.icrs.ra.to_string(unit=u.hour,sep='',pad=True)
    dec_names = ml_coords.icrs.dec.to_string(sep='',pad=True,alwayssign=True)
    vlim = [-500., 500.]*u.km/u.s
    dcoord = SkyCoord(ra=0., dec=0., unit='deg')

    # Loop on list
    didx, lidx = [], []
    print("Looping on sightlines..")
    for tt,obj in enumerate(ml_results):
        #if (tt % 100) == 0:
        #    print('tt: {:d}'.format(tt))
        # Sightline
        if use_id:
            plate, fiber = [int(spl) for spl in obj['id'].split('-')]
            idict['plate'].append(plate)
            idict['fiber'].append(fiber)

        # Systems
        for ss,syskey in enumerate(['dlas', 'subdlas']):
            for idla in obj[syskey]:
                name = 'J{:s}{:s}_z{:.3f}'.format(ra_names[tt], dec_names[tt], idla['z_dla'])
                if ss == 0:
                    isys = DLASystem(dcoord, idla['z_dla'], vlim, NHI=idla['column_density'], zem=obj['z_qso'], name=name)
                else:
                    isys = LLSSystem(dcoord, idla['z_dla'], vlim, NHI=idla['column_density'], zem=obj['z_qso'], name=name)
                isys.confidence = idla['dla_confidence']
                isys.s2n = idla['s2n']
                if use_platef:
                    isys.plate = obj['plate']
                    isys.fiber = obj['fiber']
                elif use_id:
                    isys.plate = plate
                    isys.fiber = fiber
                # Save
                if ss == 0:
                    didx.append(tt)
                    dlasystems.append(isys)
                else:
                    lidx.append(tt)
                    llssystems.append(isys)
    # Generate sightline tables
    sightlines = Table()
    sightlines['RA'] = idict['ra']
    sightlines['DEC'] = idict['dec']
    sightlines['PLATE'] = idict['plate']
    sightlines['FIBERID'] = idict['fiber']
    # Surveys
    ml_llssurvey = LLSSurvey()
    ml_llssurvey.sightlines = sightlines.copy()
    ml_llssurvey._abs_sys = llssystems
    ml_llssurvey.coords = ml_coords[np.array(lidx)]

    ml_dlasurvey = DLASurvey()
    ml_dlasurvey.sightlines = sightlines.copy()
    ml_dlasurvey._abs_sys = dlasystems
    ml_dlasurvey.coords = ml_coords[np.array(didx)]

    # Return
    return ml_llssurvey, ml_dlasurvey


def load_ml_dr12():
    # Sightlines
    dr12_sline_file = resource_filename('dla_cnn', 'catalogs/boss_dr12/DR12_sightlines.fits.gz')
    dr12_slines = Table.read(dr12_sline_file)
    # Absorbers
    dr12_abs_file = resource_filename('dla_cnn', 'catalogs/boss_dr12/DR12_DLA_SLLS.fits.gz')
    dr12_abs = Table.read(dr12_abs_file)
    # Return
    return dr12_slines, dr12_abs


def load_garnett16(orig=False):
    """ Read one of the Garnett files
    Parameters
    ----------
    orig : bool, optional
      Use the original files, i.e. prior to full publication in MNRAS

    Returns
    -------
    g16_abs : Table
      Table describing the BOSS sightlines
    """
    if orig:
        g16_abs_file = resource_filename('dla_cnn', 'catalogs/boss_dr12/DR12_DLA_garnett16_orig.fits.gz')
    else:
        g16_abs_file = resource_filename('dla_cnn', 'catalogs/boss_dr12/MNRAS/merged_g16.fits.gz')
    g16_abs = Table.read(g16_abs_file)
    return g16_abs
