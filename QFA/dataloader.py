import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from .utils import smooth
from .utils import tau as default_tau
from typing import Tuple, Callable
from yacs.config import CfgNode as CN


_lya_peak = 1215.67


def _read_npz_file(path: str)->Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    load spectra from npz files
    NOTE:
        (1) all spectra should have same wavelength grid
        (2) spectra are preprocessed as in the paper
        (3) missing pixels are denoted with -999.
    """
    file = np.load(path)
    flux, error, z = file['flux'], file['error'], float(file['z'])
    mask = (flux!=-999.)&(error!=-999.)
    file.close()
    return flux, error, mask, z


def _read_from_catalog(flux, error, mask, zqso, catalog, data_dir, num, snr_min, snr_max, z_min, z_max, num_mask, nprocs):
    catalog = pd.read_csv(catalog)
    criteria = (catalog['snr']>=snr_min) & (catalog['snr']<=snr_max) & (catalog['z']>=z_min) & (catalog['z']<=z_max) & (catalog['num_mask']<=num_mask)
    files = np.random.choice(catalog['file'][criteria].values, size=(num,), replace=(np.sum(criteria)<num))
    paths = [os.path.join(data_dir, x) for x in files]
    with multiprocessing.Pool(nprocs) as p:
        data = p.map(_read_npz_file, paths)
    for f, e, m, z in tqdm(data):
        flux.append(f)
        error.append(e)
        mask.append(m)
        zqso.append(z)


class Dataloader(object):

    def __init__(self, config: CN):
        self.wav_grid = 10**np.arange(np.log10(config.DATA.LAMMIN), np.log10(config.DATA.LAMMAX), config.DATA.LOGLAM_DELTA)
        self.Nb = np.sum(self.wav_grid<_lya_peak)
        self.Nr = len(self.wav_grid) - self.Nb

        self.batch_size = config.DATA.BATCH_SIZE

        self.flux = []
        self.error = []
        self.mask = []
        self.zabs = []
        self.zqso = []
        
        print("=> Load Data...")
        _read_from_catalog(self.flux, self.error, self.mask, self.zqso, config.DATA.CATALOG,
            config.DATA.DATA_DIR, config.DATA.DATA_NUM, config.DATA.SNR_MIN, config.DATA.SNR_MAX, config.DATA.Z_MIN,
            config.DATA.Z_MAX, config.DATA.NUM_MASK, config.DATA.NPROCS)
        
        if os.path.exists(config.DATA.VALIDATION_CATALOG) and os.path.exists(config.DATA.VALIDATION_DIR) and config.DATA.VALIDATION:
            print("=> Load Validation Data...")
            _read_from_catalog(self.flux, self.error, self.mask, self.zqso, config.DATA.VALIDATION_CATALOG,
            config.DATA.VALIDATION_DIR, config.DATA.VALIDATION_NUM, config.DATA.SNR_MIN, config.DATA.SNR_MAX,
            config.DATA.Z_MIN, config.DATA.Z_MAX, config.DATA.NUM_MASK, config.DATA.NPROCS)


        self.flux = np.array(self.flux)
        self.error = np.array(self.error)
        self.zqso = np.array(self.zqso)
        self.mask = np.array(self.mask)
        self.zabs = (self.zqso + 1).reshape(-1, 1)*self.wav_grid[:self.Nb]/1215.67 - 1


        self.cur = 0
        self._device = None
        self._tau = default_tau
        self.validation_dir = None
        self.data_size = self.flux.shape[0]
        
        s = np.hstack((np.exp(1*self._tau(self.zabs)), np.ones((self.data_size, self.Nr), dtype=float)))
        self._mu = np.sum(self.flux*s, axis=0)/np.sum(self.flux!=0., axis=0)
        self._mu = smooth(self._mu, window_len=config.TRAIN.WINDOW_LENGTH_FOR_MU)

    def have_next_batch(self):
        """
        indicate whether this dataloader have next batch

        Returns:
            sig (bool): whether this dataloader have next batch
        """
        return self.cur < self.data_size

    def next_batch(self):
        """
        next batch for this dataloader

        Returns:
            delta, error, redshift, mask (torch.tensor): batch data
        """
        start = self.cur
        end = self.cur + self.batch_size if self.cur + self.batch_size < self.data_size else self.data_size
        self.cur = end
        s = np.hstack((np.exp(-1*self._tau(self.zabs[start: end])), np.ones((-start+end, self.Nr), dtype=float)))
        return torch.tensor(self.flux[start: end]-self._mu*s, dtype=torch.float32).to(self._device),\
            torch.tensor(self.error[start: end], dtype=torch.float32).to(self._device), torch.tensor(self.zabs[start: end], dtype=torch.float32).to(self._device), \
                torch.tensor(self.mask[start: end], dtype=bool).to(self._device)

    def sample(self):
        """
        draw sample from this dataloader

        Returns:
            delta, error, redshift, mask (torch.tensor): sampled data
        """
        sig = np.random.randint(0, self.data_size, size=(self.batch_size, ))
        s = np.hstack((np.exp(-1.*self._tau(self.zabs[sig])), np.ones((self.batch_size, self.Nr), dtype=float)))
        return torch.tensor(self.flux[sig]-self._mu*s, dtype=torch.tensor32).to(self._device),\
            torch.tensor(self.error[sig], dtype=torch.float32).to(self._device), torch.tensor(self.zabs[sig], dtype=torch.float32).to(self._device), \
                torch.tensor(self.mask[sig], dtype=bool).to(self._device)

    def rewind(self):
        """
        shuffle all the data and reset the dataloader
        """
        idx = np.arange(self.data_size)
        np.random.shuffle(idx)
        self.cur = 0
        self.flux = self.flux[idx]
        self.error = self.error[idx]
        self.zqso = self.zqso[idx]
        self.zabs = self.zabs[idx]
        self.mask = self.mask[idx]

    def set_tau(self, tau:Callable[[torch.tensor, ], torch.tensor])->None:
        """
        set pre defined mean optical depth function
        """
        self._tau = tau

    def set_device(self, device: torch.device)->None:
        """
        set which device to run the model
        """
        self._device = device
    
    @property
    def mu(self):
        return self._mu