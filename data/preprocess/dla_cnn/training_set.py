""" Module to vette results against Human catalogs
  SDSS-DR5 (JXP) and BOSS (Notredaeme)
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import pdb

from scipy import interpolate


from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u
from astropy.table import Table
from astropy.io.fits import Header

from specdb.specdb import IgmSpec

from linetools import utils as ltu
from linetools.spectra.xspectrum1d import XSpectrum1D
from linetools.lists.linelist import LineList
from pyigm.surveys.dlasurvey import DLASurvey

llist = LineList('HI')

def grab_sightlines(dlasurvey=None, flg_bal=None, zmin=2.3, s2n=5., DX=0.,
                    igmsp_survey='SDSS_DR7', update_zem=True):
    """ Grab a set of sightlines without DLAs from a DLA survey
    Insist that all have spectra occur in igmspec
    Update sightline zem with igmspec zem

    Parameters
    ----------
    dlas : DLASurvey
      Usually SDSS or BOSS
    flg_bal : int, optional
      Maximum BAL flag (0=No signature, 1=Weak BAL, 2=BAL)
    s2n : float, optional
      Minimum S/N as defined in some manner
    DX : float, optional
      Restrict on DX
    zmin : float, optional
      Minimum redshift for zem
    update_zem : bool, optional
      Update zem in sightlines?

    Returns
    -------
    final : Table
      astropy Table of good sightlines
    sdict : dict
      dict describing the sightlines
    """
    #1)  REMOVE 910, 526  z=2.88; NHI=21.19
    import warnings
    warnings.warn("Someday remove 910, 526 which has a *strong* DLA")
    igmsp = IgmSpec()
    # Init
    if dlasurvey is None:
        print("Using the DR5 sample for the sightlines")
        dlasurvey = DLASurvey.load_SDSS_DR5(sample='all')
        igmsp_survey = 'SDSS_DR7'
    nsight = len(dlasurvey.sightlines)
    keep = np.array([True]*nsight)
    meta = igmsp[igmsp_survey].meta

    # Avoid DLAs
    dla_coord = dlasurvey.coord
    sl_coord = SkyCoord(ra=dlasurvey.sightlines['RA'], dec=dlasurvey.sightlines['DEC'])
    idx, d2d, d3d = match_coordinates_sky(sl_coord, dla_coord, nthneighbor=1)
    clear = d2d > 1*u.arcsec
    keep = keep & clear

    # BAL
    if flg_bal is not None:
        gd_bal = dlasurvey.sightlines['FLG_BAL'] <= flg_bal
        keep = keep & gd_bal

    # S/N
    if s2n > 0.:
        gd_s2n = dlasurvey.sightlines['S2N'] > s2n
        keep = keep & gd_s2n

    # Cut on DX
    if DX > 0.:
        gd_DX = dlasurvey.sightlines['DX'] > DX
        keep = keep & gd_DX

    # igmsp
    qso_coord = SkyCoord(ra=meta['RA_GROUP'], dec=meta['DEC_GROUP'], unit='deg')
    idxq, d2dq, d3dq = match_coordinates_sky(sl_coord, qso_coord, nthneighbor=1)
    in_igmsp = d2dq < 1*u.arcsec
    keep = keep & in_igmsp

    # Check zem and dz
    #igm_id = meta['IGM_ID'][idxq]
    #cat_rows = match_ids(igm_id, igmsp.cat['IGM_ID'])
    #zem = igmsp.cat['zem'][cat_rows]
    zem = meta['zem_GROUP'][idxq]
    dz = np.abs(zem - dlasurvey.sightlines['ZEM'])
    gd_dz = dz < 0.1
    keep = keep & gd_dz #& gd_zlim
    if zmin is not None:
        gd_zmin = zem > zmin
        keep = keep & gd_zmin #& gd_zlim
    #gd_zlim = (zem-dlasurvey.sightlines['Z_START']) > 0.1
    #pdb.set_trace()

    # Assess
    final = dlasurvey.sightlines[keep]
    #final_coords = SkyCoord(ra=final['RA'], dec=final['DEC'], unit='deg')
    #matches, meta = igmsp.meta_from_coords(final_coords, groups=['SDSS_DR7'], tol=1*u.arcsec)
    #idxq2, d2dq2, d3dq2 = match_coordinates_sky(final_coords, qso_coord, nthneighbor=1)
    #in_igmsp2 = d2dq2 < 1*u.arcsec
    #pdb.set_trace()
    sdict = {}
    sdict['n'] = len(final)
    print("We have {:d} sightlines for analysis".format(sdict['n']))

    def qck_stats(idict, tbl, istr, key):
        idict[istr+'min'] = np.min(tbl[key])
        idict[istr+'max'] = np.max(tbl[key])
        idict[istr+'median'] = np.median(tbl[key])
    qck_stats(sdict, final, 'z', 'ZEM')
    qck_stats(sdict, final, 'i', 'MAG')

    print("Min z = {:g}, Median z = {:g}, Max z = {:g}".format(sdict['zmin'], sdict['zmedian'], sdict['zmax']))

    # Return
    return final, sdict


def init_fNHI(slls=False, mix=False, high=False):
    """ Generate the interpolator for log NHI

    Returns
    -------
    fNHI : scipy.interpolate.interp1d function
    """
    from pyigm.fN.fnmodel import FNModel
    # f(N)
    fN_model = FNModel.default_model()
    # Integrate on NHI
    if slls:
        lX, cum_lX, lX_NHI = fN_model.calculate_lox(fN_model.zpivot,
                                                    19.5, NHI_max=20.3, cumul=True)
    elif high:
        lX, cum_lX, lX_NHI = fN_model.calculate_lox(fN_model.zpivot,
                                                    21.2, NHI_max=22.5, cumul=True)
    elif mix:
        lX, cum_lX, lX_NHI = fN_model.calculate_lox(fN_model.zpivot,
                                                    19.5, NHI_max=22.5, cumul=True)
    else:
        lX, cum_lX, lX_NHI = fN_model.calculate_lox(fN_model.zpivot,
                                                20.3, NHI_max=22.5, cumul=True)
    # Interpolator
    cum_lX /= cum_lX[-1] # Normalize
    fNHI = interpolate.interp1d(cum_lX, lX_NHI,
                                bounds_error=False,fill_value=lX_NHI[0])
    return fNHI


def insert_dlas(spec, zem, fNHI=None, rstate=None, slls=False,
                mix=False, high=False, low_s2n=False, noise_boost=4.):
    """ Insert a DLA into input spectrum
    Also adjusts the noise
    Will also add noise 'everywhere' if requested
    Parameters
    ----------
    spec
    fNHI
    rstate
    low_s2n : bool, optional
      Reduce the S/N everywhere.  By a factor of noise_boost
    noise_boost : float, optional
      Factor to *increase* the noise by

    Returns
    -------
    final_spec : XSpectrum1D  
    dlas : list
      List of DLAs inserted

    """
    from pyigm.fN import dla as pyi_fd
    from pyigm.abssys.dla import DLASystem
    from pyigm.abssys.lls import LLSSystem
    from pyigm.abssys.utils import hi_model

    # Init
    if rstate is None:
        rstate = np.random.RandomState()
    if fNHI is None:
        fNHI = init_fNHI(slls=slls, mix=mix, high=high)

    # Allowed redshift placement
    ## Cut on zem and 910A rest-frame
    zlya = spec.wavelength.value/1215.67 - 1
    dz = np.roll(zlya,-1)-zlya
    dz[-1] = dz[-2]
    gdz = (zlya < zem) & (spec.wavelength > 910.*u.AA*(1+zem))

    # l(z) -- Uses DLA for SLLS too which is fine
    lz = pyi_fd.lX(zlya[gdz], extrap=True, calc_lz=True)
    cum_lz = np.cumsum(lz*dz[gdz])
    tot_lz = cum_lz[-1]
    fzdla = interpolate.interp1d(cum_lz/tot_lz, zlya[gdz],
                                 bounds_error=False,fill_value=np.min(zlya[gdz]))#

    # n DLA
    nDLA = 0
    while nDLA == 0:
        nval = rstate.poisson(tot_lz, 100)
        gdv = nval > 0
        if np.sum(gdv) == 0:
            continue
        else:
            nDLA = nval[np.where(gdv)[0][0]]

    # Generate DLAs
    dlas = []
    for jj in range(nDLA):
        # Random z
        zabs = float(fzdla(rstate.random_sample()))
        # Random NHI
        NHI = float(fNHI(rstate.random_sample()))
        if (slls or mix):
            dla = LLSSystem((0.,0), zabs, None, NHI=NHI)
        else:
            dla = DLASystem((0.,0), zabs, (None,None), NHI)
        dlas.append(dla)

    # Insert
    vmodel, _ = hi_model(dlas, spec, fwhm=3., llist=llist)
    # Add noise
    rand = rstate.randn(spec.npix)
    noise = rand * spec.sig * (1-vmodel.flux.value)
    # More noise??
    if low_s2n:
        rand2 = rstate.randn(spec.npix)
        more_noise = noise_boost * rand2 * spec.sig
        noise += more_noise
    else:
        s2n_boost=1.

    final_spec = XSpectrum1D.from_tuple((vmodel.wavelength,
                                         spec.flux.value*vmodel.flux.value+noise,
                                         noise_boost*spec.sig))

    # Return
    return final_spec, dlas


def make_set(ntrain, slines, outroot=None, tol=1*u.arcsec, igmsp_survey='SDSS_DR7',
             frac_without=0., seed=1234, zmin=None, zmax=4.5, high=False,
             slls=False, mix=False, low_s2n=False):
    """ Generate a training set

    Parameters
    ----------
    ntrain : int
      Number of training sightlines to generate
    slines : Table
      Table of sightlines without DLAs (usually from SDSS or BOSS)
    igmsp_survey : str, optional
      Dataset name for spectra
    frac_without : float, optional
      Fraction of sightlines (on average) without a DLA
    seed : int, optional
    outroot : str, optional
      Root for output filenames
        root+'.fits' for spectra
        root+'.json' for DLA info
    zmin : float, optional
      Minimum redshift for training; defaults to min(slines['ZEM'])
    zmax : float, optional
      Maximum redshift to train on
    mix : bool, optional
      Mix of SLLS and DLAs
    low_s2n : bool, optional
      Reduce the S/N artificially, i.e. add noise

    Returns
    -------

    """
    from linetools.spectra.utils import collate

    # Init and checks
    igmsp = IgmSpec()
    assert igmsp_survey in igmsp.groups
    rstate = np.random.RandomState(seed)
    rfrac = rstate.random_sample(ntrain)
    if zmin is None:
        zmin = np.min(slines['ZEM'])
    rzem = zmin + rstate.random_sample(ntrain)*(zmax-zmin)
    fNHI = init_fNHI(slls=slls, mix=mix, high=high)

    all_spec = []
    full_dict = {}
    # Begin looping
    for qq in range(ntrain):
        print("qq = {:d}".format(qq))
        full_dict[qq] = {}
        # Grab sightline
        isl = np.argmin(np.abs(slines['ZEM']-rzem[qq]))
        full_dict[qq]['sl'] = isl  # sightline
        specl, meta = igmsp.spectra_from_coord((slines['RA'][isl], slines['DEC'][isl]),
                                           groups=['SDSS_DR7'], tol=tol, verbose=False)
        assert len(meta) == 1
        spec = specl
        # Meta data for header
        mdict = {}
        for key in meta.keys():
            mdict[key] = meta[key][0]
        mhead = Header(mdict)
        # Clear?
        if rfrac[qq] < frac_without:
            spec.meta['headers'][0] = mdict.copy() #mhead
            all_spec.append(spec)
            full_dict[qq]['nDLA'] = 0
            continue
        # Insert at least one DLA
        spec, dlas = insert_dlas(spec, mhead['zem_GROUP'], rstate=rstate,
                                 fNHI=fNHI, slls=slls, mix=mix, high=high,
                                 low_s2n=low_s2n)
        spec.meta['headers'][0] = mdict.copy() #mhead
        all_spec.append(spec)
        full_dict[qq]['nDLA'] = len(dlas)
        for kk,dla in enumerate(dlas):
            full_dict[qq][kk] = {}
            full_dict[qq][kk]['NHI'] = dla.NHI
            full_dict[qq][kk]['zabs'] = dla.zabs

    # Generate one object
    final_spec = collate(all_spec)
    # Write?
    if outroot is not None:
        # Spectra
        final_spec.write_to_hdf5(outroot+'.hdf5')
        # Dict -> JSON
        gdict = ltu.jsonify(full_dict)
        ltu.savejson(outroot+'.json', gdict, overwrite=True)#, easy_to_read=True)
    # Return
    return final_spec, full_dict


def training_prod(seed, nruns, nsline, nproc=10, outpath='./', slls=False):
    """ Perform a full production run of training sightlines

    Parameters
    ----------
    seed
    nsline
    outpath

    Returns
    -------

    """
    from subprocess import Popen
    rstate = np.random.RandomState(seed)
    # Generate individual seeds
    seeds = np.round(100000*rstate.random_sample(nruns)).astype(int)

    # Start looping on processor
        # Loop on the systems
    nrun = -1
    while(nrun < nruns):
        proc = []
        for ss in range(nproc):
            nrun += 1
            if nrun == nruns:
                break
            # Run
            script = ['./scripts/dlaml_trainingset.py', str(seeds[nrun]), str(nsline), str(outpath)]
            if slls:
                script = ['./scripts/dlaml_trainingset.py', str(seeds[nrun]), str(nsline), str(outpath), str('--slls')]
            proc.append(Popen(script))
        exit_codes = [p.wait() for p in proc]


def write_sdss_sightlines():
    """ Writes the SDSS DR5 sightlines that have no (or very few) DLAs
    Returns
    -------
    None : Writes to Dropbox

    """
    import os
    import h5py
    outfile=os.getenv('DROPBOX_DIR')+'/MachineLearning/DR5/SDSS_DR5_noDLAs.hdf5'
    # Load
    sdss = DLASurvey.load_SDSS_DR5(sample='all')
    slines, sdict = grab_sightlines(sdss, flg_bal=0)
    coords = SkyCoord(ra=slines['RA'], dec=slines['DEC'], unit='deg')
    # Load spectra -- RA/DEC in igmsp is not identical to RA_GROUP, DEC_GROUP in SDSS_DR7
    igmsp = IgmSpec()
    sdss_meta = igmsp['SDSS_DR7'].meta
    qso_coord = SkyCoord(ra=sdss_meta['RA_GROUP'], dec=sdss_meta['DEC_GROUP'], unit='deg')
    idxq, d2dq, d3dq = match_coordinates_sky(coords, qso_coord, nthneighbor=1)
    in_igmsp = d2dq < 1*u.arcsec # Check
    # Cut meta
    cut_meta = sdss_meta[idxq[in_igmsp]]
    assert len(slines) == len(cut_meta)
    # Grab
    spectra = igmsp['SDSS_DR7'].spec_from_meta(cut_meta)
    # Write
    hdf = h5py.File(outfile,'w')
    spectra.write_to_hdf5(outfile, hdf5=hdf, clobber=True, fill_val=0.)
    # Add table (meta is already used)
    hdf['cut_meta'] = cut_meta
    hdf.close()


def save_np_dataset(save_file, data):
    """
    Write input dataset (typically training or test) to a Numpy file

    Args:
        save_file (str):
        data (dict): dict holding the flux and labels for the dataset

    """
    print("Writing %s.npy to disk" % save_file)
    # np.save(save_file+".npy", data['flux'])
    # data['flux'] = None
    # print "Writing %s.pickle to disk" % save_file
    # with gzip.GzipFile(filename=save_file+".pickle", mode='wb', compresslevel=2) as f:
    #     pickle.dump([data], f, protocol=-1)
    np.savez_compressed(save_file,
                        flux=data['flux'],
                        labels_classifier=data['labels_classifier'],
                        labels_offset=data['labels_offset'],
                        col_density=data['col_density'])


# Receives data in the tuple form returned from split_sightline_into_samples:
# (fluxes_matrix, classification, offsets_array, column_density)
# Returns indexes of pos & neg samples that are 50% positive and 50% negative and no boarder
def select_samples_50p_pos_neg(classification):
    """
    For a given sightline, generate the indices for DLAs and for without
    Split 50/50 to have equal representation

    Parameters
    ----------
    classification: np.ndarray
        Array of classification values.  1=DLA; 0=Not; -1=not analyzed

    Returns
    -------
    idx: np.ndarray
        positive + negative indices

    """
    #classification = data[1]
    num_pos = np.sum(classification==1, dtype=np.float64)
    num_neg = np.sum(classification==0, dtype=np.float64)
    n_samples = int(min(num_pos, num_neg))

    r = np.random.permutation(len(classification))

    pos_ixs = r[classification[r]==1][0:n_samples]
    neg_ixs = r[classification[r]==0][0:n_samples]
    # num_total = data[0].shape[0]
    # ratio_neg = num_pos / num_neg

    # pos_mask = classification == 1      # Take all positive samples

    # neg_ixs_by_ratio = np.linspace(1,num_total-1,round(ratio_neg*num_total), dtype=np.int32) # get all samples by ratio
    # neg_mask = np.zeros((num_total),dtype=np.bool) # create a 0 vector of negative samples
    # neg_mask[neg_ixs_by_ratio] = True # set the vector to positives, selecting for the appropriate ratio across the whole sightline
    # neg_mask[pos_mask] = False # remove previously positive samples from the set
    # neg_mask[classification == -1] = False # remove border samples from the set, what remains is still in the right ratio

    # return pos_mask | neg_mask
    return np.hstack((pos_ixs,neg_ixs))

