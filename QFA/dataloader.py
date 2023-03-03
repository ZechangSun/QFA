import os
import torch
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
import multiprocessing
from .utils import smooth
from .utils import tau as taufunc
from typing import Tuple, Callable, List
from functools import partial
from yacs.config import CfgNode as CN


_lya_peak = 1215.67


def _read_npz_file(path: str)->Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    load spectra from npz file
    NOTE:
        (1) all spectra should have same wavelength grid
        (2) spectra are preprocessed as in the paper
        (3) missing pixels are denoted with -999.
    """
    file = np.load(path)
    flux, error, z = file['flux'], file['error'], float(file['z'])
    mask = (flux!=-999.)&(error!=-999.)
    file.close()
    return flux, error, mask, z, path


def _read_npz_files(flux: List[np.ndarray], error: List[np.ndarray], mask: List[np.ndarray], zqso: List[np.ndarray], pathlist: List[np.ndarray], paths: str, nprocs: int)->Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    load spectra from npz files
    """
    with multiprocessing.Pool(nprocs) as p:
        data = p.map(_read_npz_file, paths)
    for f, e, m, z, p in tqdm(data):
        flux.append(f)
        error.append(e)
        mask.append(m)
        zqso.append(z)
        pathlist.append(p)
    

def _read_from_catalog(flux, error, mask, zqso, pathlist, catalog, data_dir, num, snr_min, snr_max, z_min, z_max, num_mask, nprocs, output_dir, prefix='train'):
    catalog = pd.read_csv(catalog)
    criteria = (catalog['snr']>=snr_min) & (catalog['snr']<=snr_max) & (catalog['z']>=z_min) & (catalog['z']<=z_max) & (catalog['num_mask']<=num_mask)
    files = np.random.choice(catalog['file'][criteria].values, size=(num,), replace=(np.sum(criteria)<num))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    pd.Series(files).to_csv(os.path.join(output_dir, f'{prefix}-catalog.csv'), header=False, index=False)
    paths = [os.path.join(data_dir, x) for x in files]
    _read_npz_files(flux, error, mask, zqso, pathlist, paths, nprocs)


class Dataloader(object):

    def __init__(self, config: CN):
        self.wav_grid = 10**np.arange(np.log10(config.DATA.LAMMIN), np.log10(config.DATA.LAMMAX), config.DATA.LOGLAM_DELTA)
        self.Nb = np.sum(self.wav_grid<_lya_peak)
        self.Nr = len(self.wav_grid) - self.Nb
        self.type = config.TYPE

        self.batch_size = config.DATA.BATCH_SIZE

        self.flux = []
        self.error = []
        self.mask = []
        self.zabs = []
        self.zqso = []
        self.pathlist = []

        if self.type == 'train':
            print("=> Load Data...")
            _read_from_catalog(self.flux, self.error, self.mask, self.zqso, self.pathlist, config.DATA.CATALOG,
                config.DATA.DATA_DIR, config.DATA.DATA_NUM, config.DATA.SNR_MIN, config.DATA.SNR_MAX, config.DATA.Z_MIN,
                config.DATA.Z_MAX, config.DATA.NUM_MASK, config.DATA.NPROCS, config.DATA.OUTPUT_DIR, 'train')
        
            if os.path.exists(config.DATA.VALIDATION_CATALOG) and os.path.exists(config.DATA.VALIDATION_DIR) and config.DATA.VALIDATION:
                print("=> Load Validation Data...")
                _read_from_catalog(self.flux, self.error, self.mask, self.zqso, self.pathlist, config.DATA.VALIDATION_CATALOG,
                config.DATA.VALIDATION_DIR, config.DATA.VALIDATION_NUM, config.DATA.SNR_MIN, config.DATA.SNR_MAX,
                config.DATA.Z_MIN, config.DATA.Z_MAX, config.DATA.NUM_MASK, config.DATA.NPROCS, config.DATA.OUTPUT_DIR, 'validation')
        
        elif self.type == 'predict':
            print("=> Load Data...")
            paths = pd.read_csv(config.DATA.CATALOG).values.squeeze()
            paths = list(map(lambda x: os.path.join(config.DATA.DATA_DIR, x), paths))
            _read_npz_files(self.flux, self.error, self.mask, self.zqso, self.pathlist, paths, config.DATA.NPROCS)
        
        else:
            raise NotImplementedError("TYPE should be in ['train', 'test']!")


        self.flux = np.array(self.flux)
        self.error = np.array(self.error)
        self.zqso = np.array(self.zqso)
        self.mask = np.array(self.mask)
        self.pathlist = np.array(self.pathlist)
        self.zabs = (self.zqso + 1).reshape(-1, 1)*self.wav_grid[:self.Nb]/1215.67 - 1


        self.cur = 0
        self._device = None
        self._tau = partial(taufunc, which=config.MODEL.TAU)
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
        if self.type == 'test': warnings.warn('dataloader is in test mode...')
        return self.cur < self.data_size

    def next_batch(self):
        """
        next batch for this dataloader

        Returns:
            delta, error, redshift, mask (torch.tensor): batch data
        """
        if self.type == 'test': warnings.warn('dataloader is in test mode...')
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
        if self.type == 'test': warnings.warn('dataloader is in test mode...')
        sig = np.random.randint(0, self.data_size, size=(self.batch_size, ))
        s = np.hstack((np.exp(-1.*self._tau(self.zabs[sig])), np.ones((self.batch_size, self.Nr), dtype=float)))
        return torch.tensor(self.flux[sig]-self._mu*s, dtype=torch.tensor32).to(self._device),\
            torch.tensor(self.error[sig], dtype=torch.float32).to(self._device), torch.tensor(self.zabs[sig], dtype=torch.float32).to(self._device), \
                torch.tensor(self.mask[sig], dtype=bool).to(self._device)

    def rewind(self):
        """
        shuffle all the data and reset the dataloader
        """
        if self.type == 'test': warnings.warn('dataloader is in test mode...')
        idx = np.arange(self.data_size)
        np.random.shuffle(idx)
        self.cur = 0
        self.flux = self.flux[idx]
        self.error = self.error[idx]
        self.zqso = self.zqso[idx]
        self.zabs = self.zabs[idx]
        self.mask = self.mask[idx]
        self.pathlist = self.pathlist[idx]

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
    
    def __len__(self):
        return len(self.flux)

    def __getitem__(self, idx):
        return torch.tensor(self.flux[idx], dtype=torch.float32).to(self._device),\
            torch.tensor(self.error[idx], dtype=torch.float32).to(self._device), torch.tensor(self.zabs[idx], dtype=torch.float32).to(self._device), \
                torch.tensor(self.mask[idx], dtype=bool).to(self._device), self.pathlist[idx]

    @property
    def mu(self):
        return self._mu