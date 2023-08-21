""" Module of routines related to DLA catalogs
However I/O is in io.py
"""
from __future__ import print_function, absolute_import, division, unicode_literals

from pkg_resources import resource_filename

import os
import numpy as np

from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u

from linetools import utils as ltu

def generate_boss_tables():
    """
    Returns
    -------

    """
    # Load JSON file
    dr12_json = resource_filename('dla_cnn', 'catalogs/boss_dr12/predictions_DR12.json')
    dr12 = ltu.loadjson(dr12_json)

    # Load Garnett Table 2 for BALs
    tbl2_garnett_file = '/media/xavier/ExtraDrive2/Projects/ML_DLA_results/garnett16/ascii_catalog/table2.dat'
    tbl2_garnett = Table.read(tbl2_garnett_file, format='cds')
    tbl2_garnett_coords = SkyCoord(ra=tbl2_garnett['RAdeg'], dec=tbl2_garnett['DEdeg'], unit='deg')


    # Parse into tables
    s_plates = []
    s_fibers = []
    s_mjds = []
    s_ra = []
    s_dec = []
    s_zem = []

    a_zabs = []
    a_NHI = []
    a_sigNHI = []
    a_conf = []
    a_plates = []
    a_fibers = []
    a_mjds = []
    a_ra = []
    a_dec = []
    a_zem = []
    for sline in dr12:
        # Plate/fiber
        plate, mjd, fiber = [int(spl) for spl in sline['id'].split('-')]
        s_plates.append(plate)
        s_mjds.append(mjd)
        s_fibers.append(fiber)
        # RA/DEC/zem
        s_ra.append(sline['ra'])
        s_dec.append(sline['dec'])
        s_zem.append(sline['z_qso'])
        # DLAs/SLLS
        for abs in sline['dlas']+sline['subdlas']:
            a_plates.append(plate)
            a_mjds.append(mjd)
            a_fibers.append(fiber)
            # RA/DEC/zem
            a_ra.append(sline['ra'])
            a_dec.append(sline['dec'])
            a_zem.append(sline['z_qso'])
            # Absorber
            a_zabs.append(abs['z_dla'])
            a_NHI.append(abs['column_density'])
            a_sigNHI.append(abs['std_column_density'])
            a_conf.append(abs['dla_confidence'])
    # Sightline tables
    sline_tbl = Table()
    sline_tbl['Plate'] = s_plates
    sline_tbl['Fiber'] = s_fibers
    sline_tbl['MJD'] = s_mjds
    sline_tbl['RA'] = s_ra
    sline_tbl['DEC'] = s_dec
    sline_tbl['zem'] = s_zem

    # Match and fill BAL flag
    dr12_sline_coord = SkyCoord(ra=sline_tbl['RA'], dec=sline_tbl['DEC'], unit='deg')
    sline_tbl['flg_BAL'] = -1
    idx, d2d, d3d = match_coordinates_sky(dr12_sline_coord, tbl2_garnett_coords, nthneighbor=1)
    in_garnett = d2d < 1*u.arcsec  # Check
    sline_tbl['flg_BAL'][in_garnett] = tbl2_garnett['f_BAL'][idx[in_garnett]]
    print("There were {:d} DR12 sightlines not in Garnett".format(np.sum(~in_garnett)))

    # Write
    dr12_sline = resource_filename('dla_cnn', 'catalogs/boss_dr12/DR12_sightlines.fits')
    sline_tbl.write(dr12_sline, overwrite=True)
    print("Wrote {:s}".format(dr12_sline))

    # DLA/SLLS table
    abs_tbl = Table()
    abs_tbl['Plate'] = a_plates
    abs_tbl['Fiber'] = a_fibers
    abs_tbl['MJD'] = a_mjds
    abs_tbl['RA'] = a_ra
    abs_tbl['DEC'] = a_dec
    abs_tbl['zem'] = a_zem
    #
    abs_tbl['zabs'] = a_zabs
    abs_tbl['NHI'] = a_NHI
    abs_tbl['sigNHI'] = a_sigNHI
    abs_tbl['conf'] = a_conf
    # BAL
    dr12_abs_coord = SkyCoord(ra=abs_tbl['RA'], dec=abs_tbl['DEC'], unit='deg')
    idx, d2d, d3d = match_coordinates_sky(dr12_abs_coord, tbl2_garnett_coords, nthneighbor=1)
    in_garnett = d2d < 1*u.arcsec  # Check
    abs_tbl['flg_BAL'] = -1
    abs_tbl['flg_BAL'][in_garnett] = tbl2_garnett['f_BAL'][idx[in_garnett]]
    abs_tbl['SNR'] = 0.
    abs_tbl['SNR'][in_garnett] = tbl2_garnett['SNRSpec'][idx[in_garnett]]
    print("There were {:d} DR12 absorbers not covered by Garnett".format(np.sum(~in_garnett)))

    dr12_abs = resource_filename('dla_cnn', 'catalogs/boss_dr12/DR12_DLA_SLLS.fits')
    abs_tbl.write(dr12_abs, overwrite=True)
    print("Wrote {:s}".format(dr12_abs))

    # Garnett
    ml_path = os.getenv('PROJECT_ML')
    g16_dlas = Table.read(ml_path + '/garnett16/ascii_catalog/table3.dat', format='cds')
    tbl3_garnett_coords = SkyCoord(ra=g16_dlas['RAdeg'], dec=g16_dlas['DEdeg'], unit='deg')
    idx, d2d, d3d = match_coordinates_sky(tbl3_garnett_coords, tbl2_garnett_coords, nthneighbor=1)
    in_garnett = d2d < 1*u.arcsec  # Check
    g16_dlas['flg_BAL'] = -1
    g16_dlas['flg_BAL'][in_garnett] = tbl2_garnett['f_BAL'][idx[in_garnett]]
    g16_dlas['SNR'] = 0.
    g16_dlas['SNR'][in_garnett] = tbl2_garnett['SNRSpec'][idx[in_garnett]]
    g16_outfile = resource_filename('dla_cnn', 'catalogs/boss_dr12/DR12_DLA_garnett16.fits')
    g16_dlas.write(g16_outfile, overwrite=True)
    print("Wrote {:s}".format(g16_outfile))


def match_boss_catalogs(dr12_dla, g16_dlas, dztoler=0.015, reverse=False):
    """ Match our ML catalog against G16 or vice-versa

    Parameters
    ----------
    dr12_dla
    g16_dlas
    dztoler
    reverse

    Returns
    -------

    """
    # Indices
    dr12_to_g16 = np.zeros(len(dr12_dla)).astype(int) -1
    # Search around
    if reverse:
        dr12_dla_coords = SkyCoord(ra=dr12_dla['RAdeg'], dec=dr12_dla['DEdeg'], unit='deg')
        g16_coord = SkyCoord(ra=g16_dlas['RA'], dec=g16_dlas['DEC'], unit='deg')
    else:
        dr12_dla_coords = SkyCoord(ra=dr12_dla['RA'], dec=dr12_dla['DEC'], unit='deg')
        g16_coord = SkyCoord(ra=g16_dlas['RAdeg'], dec=g16_dlas['DEdeg'], unit='deg')
    idx_g16, idx_dr12, d2d, d3d = dr12_dla_coords.search_around_sky(g16_coord, 1*u.arcsec)

    # Loop to match
    for kk,idx in enumerate(idx_dr12):
        if reverse:
            dz = np.abs(dr12_dla['z_DLA'][idx] - g16_dlas['zabs'][idx_g16[kk]])
        else:
            dz = np.abs(dr12_dla['zabs'][idx] - g16_dlas['z_DLA'][idx_g16[kk]])
        if dz < dztoler:
            dr12_to_g16[idx] = idx_g16[kk]
    # Return
    return dr12_to_g16

def main(flg_cat):
    import os

    # BOSS tables
    if (flg_cat & 2**0):
        generate_boss_tables()

# Test
if __name__ == '__main__':
    flg_cat = 0
    flg_cat += 2**0   # BOSS Tables
    #flg_cat += 2**7   # Training set of high NHI systems

    main(flg_cat)
