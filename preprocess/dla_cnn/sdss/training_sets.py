""" Training sets for SDSS/BOSS.  Parks et al. 2018"""

import numpy as np

from dla_cnn.data_model import Sightline
from dla_cnn.data_model import Id_DR12
from dla_cnn.data_model import Dla

from dla_cnn.training_set import save_np_dataset

def preprocess_data_from_dr9(kernel=400, stride=3, pos_sample_kernel_percent=0.3,
                             train_keys_csv="../data/dr9_train_set.csv",
                             test_keys_csv="../data/dr9_test_set.csv"):
    """
    Custom training sets for SDSS DR9

    Args:
        kernel:
        stride:
        pos_sample_kernel_percent:
        train_keys_csv:
        test_keys_csv:

    Returns:

    """
    dr9_train = np.genfromtxt(train_keys_csv, delimiter=',')
    dr9_test = np.genfromtxt(test_keys_csv, delimiter=',')

    # Dedup ---(there aren't any in dr9_train, so skipping for now)
    # dr9_train_keys = np.vstack({tuple(row) for row in dr9_train[:,0:3]})

    sightlines_train = [Sightline(Id_DR12(s[0],s[1],s[2]),[Dla(s[3],s[4])]) for s in dr9_train]
    sightlines_test  = [Sightline(Id_DR12(s[0],s[1],s[2]),[Dla(s[3],s[4])]) for s in dr9_test]

    prepare_localization_training_set(kernel, stride, pos_sample_kernel_percent,
                                      sightlines_train, sightlines_test)


def prepare_localization_training_set(ids_train, ids_test,
                                      train_save_file="../data/localize_train.npy",
                                      test_save_file="../data/localize_test.npy",
                                      ignore_sightline_markers={}):
    """
    Build a Training set

    Args:
        ids_train:
        ids_test:
        train_save_file:
        test_save_file:
        ignore_sightline_markers:

    Returns:

    """
    num_cores = multiprocessing.cpu_count() - 1
    p = Pool(num_cores, maxtasksperchild=10)  # a thread pool we'll reuse

    # Training data
    with Timer(disp="read_sightlines"):
        sightlines_train = p.map(read_sightline, ids_train)
        # add the ignore markers to the sightline
        for s in sightlines_train:
            if hasattr(s.id, 'sightlineid') and s.id.sightlineid >= 0:
                s.data_markers = ignore_sightline_markers[s.id.sightlineid] if ignore_sightline_markers.has_key(
                    s.id.sightlineid) else []
    with Timer(disp="split_sightlines_into_samples"):
        data_split = p.map(split_sightline_into_samples, sightlines_train)
    with Timer(disp="select_samples_50p_pos_neg"):
        sample_masks = p.map(select_samples_50p_pos_neg, data_split)
    with Timer(disp="zip and stack"):
        zip_data_masks = zip(data_split, sample_masks)
        data_train = {}
        data_train['flux'] = np.vstack([d[0][m] for d, m in zip_data_masks])
        data_train['labels_classifier'] = np.hstack([d[1][m] for d, m in zip_data_masks])
        data_train['labels_offset'] = np.hstack([d[2][m] for d, m in zip_data_masks])
        data_train['col_density'] = np.hstack([d[3][m] for d, m in zip_data_masks])
    with Timer(disp="save train data files"):
        save_np_dataset(train_save_file, data_train)

    # Same for test data if it exists
    if len(ids_test) > 0:
        sightlines_test = p.map(read_sightline, ids_test)
        data_split = map(split_sightline_into_samples, sightlines_test)
        sample_masks = map(select_samples_50p_pos_neg, data_split)
        zip_data_masks = zip(data_split, sample_masks)
        data_test = {}
        data_test['flux'] = np.vstack([d[0][m] for d, m in zip_data_masks])
        data_test['labels_classifier'] = np.hstack([d[1][m] for d, m in zip_data_masks])
        data_test['labels_offset'] = np.hstack([d[2][m] for d, m in zip_data_masks])
        data_test['col_density'] = np.hstack([d[3][m] for d, m in zip_data_masks])
        save_np_dataset(test_save_file, data_test)


def split_sightline_into_samples(sightline,
                                 kernel=400, pos_sample_kernel_percent=0.3):
    lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
    samplerangepx = int(kernel*pos_sample_kernel_percent/2) #60
    kernelrangepx = int(kernel/2) #200
    ix_dlas = [(np.abs(lam[ix_dla_range]-dla.central_wavelength).argmin()) for dla in sightline.dlas]
    coldensity_dlas = [dla.col_density for dla in sightline.dlas]       # column densities matching ix_dlas

    # FLUXES - Produce a 1748x400 matrix of flux values
    fluxes_matrix = np.vstack(map(lambda f,r:f[r-kernelrangepx:r+kernelrangepx],
                                  zip(itertools.repeat(sightline.flux), np.nonzero(ix_dla_range)[0])))

    # CLASSIFICATION (1 = positive sample, 0 = negative sample, -1 = border sample not used
    # Start with all samples negative
    classification = np.zeros((REST_RANGE[2]), dtype=np.float32)
    # overlay samples that are too close to a known DLA, write these for all DLAs before overlaying positive sample 1's
    for ix_dla in ix_dlas:
        classification[ix_dla-samplerangepx*2:ix_dla+samplerangepx*2+1] = -1
        # Mark out Ly-B areas
        lyb_ix = sightline.get_lyb_index(ix_dla)
        classification[lyb_ix-samplerangepx:lyb_ix+samplerangepx+1] = -1
    # mark out bad samples from custom defined markers
    for marker in sightline.data_markers:
        assert marker.marker_type == Marker.IGNORE_FEATURE              # we assume there are no other marker types for now
        ixloc = np.abs(lam_rest - marker.lam_rest_location).argmin()
        classification[ixloc-samplerangepx:ixloc+samplerangepx+1] = -1
    # overlay samples that are positive
    for ix_dla in ix_dlas:
        classification[ix_dla-samplerangepx:ix_dla+samplerangepx+1] = 1

    # OFFSETS & COLUMN DENSITY
    offsets_array = np.full([REST_RANGE[2]], np.nan, dtype=np.float32)     # Start all NaN markers
    column_density = np.full([REST_RANGE[2]], np.nan, dtype=np.float32)
    # Add DLAs, this loop will work from the DLA outward updating the offset values and not update it
    # if it would overwrite something set by another nearby DLA
    for i in range(int(samplerangepx+1)):
        for ix_dla,j in zip(ix_dlas,range(len(ix_dlas))):
            offsets_array[ix_dla+i] = -i if np.isnan(offsets_array[ix_dla+i]) else offsets_array[ix_dla+i]
            offsets_array[ix_dla-i] =  i if np.isnan(offsets_array[ix_dla-i]) else offsets_array[ix_dla-i]
            column_density[ix_dla+i] = coldensity_dlas[j] if np.isnan(column_density[ix_dla+i]) else column_density[ix_dla+i]
            column_density[ix_dla-i] = coldensity_dlas[j] if np.isnan(column_density[ix_dla-i]) else column_density[ix_dla-i]
    offsets_array = np.nan_to_num(offsets_array)
    column_density = np.nan_to_num(column_density)

    # fluxes is 1748x400 of fluxes
    # classification is 1 / 0 / -1 for DLA/nonDLA/border
    # offsets_array is offset
    return fluxes_matrix, classification, offsets_array, column_density


def main(flg_tst, sdss=None, ml_survey=None):
    import os

    # Sightlines
    flg_tst = int(flg_tst)
    if (flg_tst % 2**1) >= 2**0:
        if sdss is None:
            sdss = DLASurvey.load_SDSS_DR5(sample='all')
        slines, sdict = grab_sightlines(sdss, flg_bal=0)

    # Test case of 100 sightlines
    if (flg_tst % 2**2) >= 2**1:
        # Make training set
        _, _ = make_set(100, slines, outroot='results/training_100')

    # Production runs
    if (flg_tst % 2**3) >= 2**2:
        #training_prod(123456, 5, 10, outpath=os.getenv('DROPBOX_DIR')+'/MachineLearning/DLAs/')  # TEST
        #training_prod(123456, 10, 500, outpath=os.getenv('DROPBOX_DIR')+'/MachineLearning/DLAs/')  # TEST
        training_prod(12345, 10, 5000, outpath=os.getenv('DROPBOX_DIR')+'/MachineLearning/DLAs/')

    # Production runs -- 100k more
    if (flg_tst % 2**4) >= 2**3:
        # python src/training_set.py
        training_prod(22345, 10, 10000, outpath=os.getenv('DROPBOX_DIR')+'/MachineLearning/DLAs/')

    # Production runs -- 100k more
    if flg_tst & (2**4):
        # python src/training_set.py
        if False:
            if sdss is None:
                sdss = DLASurvey.load_SDSS_DR5(sample='all')
            slines, sdict = grab_sightlines(sdss, flg_bal=0)
            _, _ = make_set(100, slines, outroot='results/slls_training_100',slls=True)
        #training_prod(22343, 10, 100, slls=True, outpath=os.getenv('DROPBOX_DIR')+'/MachineLearning/SLLSs/')
        training_prod(22343, 10, 5000, slls=True, outpath=os.getenv('DROPBOX_DIR')+'/MachineLearning/SLLSs/')

    # Mixed systems for testing
    if flg_tst & (2**5):
        # python src/training_set.py
        if sdss is None:
            sdss = DLASurvey.load_SDSS_DR5(sample='all')
        slines, sdict = grab_sightlines(sdss, flg_bal=0)
        ntrials = 10000
        seed=23559
        _, _ = make_set(ntrials, slines, seed=seed, mix=True,
                        outroot=os.getenv('DROPBOX_DIR')+'/MachineLearning/Mix/mix_test_{:d}_{:d}'.format(seed,ntrials))

    # DR5 DLA-free sightlines
    if flg_tst & (2**6):
        write_sdss_sightlines()

    # High NHI systems for testing
    if flg_tst & (2**7):
        # python src/training_set.py
        if sdss is None:
            sdss = DLASurvey.load_SDSS_DR5(sample='all')
        slines, sdict = grab_sightlines(sdss, flg_bal=0)
        ntrials = 20000
        seed=83559
        _, _ = make_set(ntrials, slines, seed=seed, high=True,
                        outroot=os.getenv('DROPBOX_DIR')+'/MachineLearning/HighNHI/high_train_{:d}_{:d}'.format(seed,ntrials))

    # Low S/N
    if flg_tst & (2**8):
        # python src/training_set.py
        if sdss is None:
            sdss = DLASurvey.load_SDSS_DR5(sample='all')
        slines, sdict = grab_sightlines(sdss, flg_bal=0)
        ntrials = 10000
        seed=83557
        _, _ = make_set(ntrials, slines, seed=seed, low_s2n=True,
                        outroot=os.getenv('DROPBOX_DIR')+'/MachineLearning/LowS2N/lows2n_train_{:d}_{:d}'.format(seed,ntrials))

# Test
if __name__ == '__main__':

    import sys
    if len(sys.argv) == 1:
        # Run from above src/
        #  I.e.   python src/training_set.py
        flg_tst = 0
        #flg_tst += 2**0   # Grab sightlines
        #flg_tst += 2**1   # First 100
        #flg_tst += 2**2   # Production run of training - fixed
        #flg_tst += 2**3   # Another production run of training - fixed seed
        #flg_tst += 2**4   # A production run with SLLS
        #flg_tst += 2**5   # A test run with a mix of SLLS and DLAs
        #flg_tst += 2**6   # Write SDSS DR5 sightlines without DLAs
        #flg_tst += 2**7   # Training set of high NHI systems
        flg_tst += 2**8   # Low S/N
    else:
        flg_tst = sys.argv[1]

    main(flg_tst)
