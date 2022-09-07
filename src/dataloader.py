import numpy as np
import torch
import multiprocessing
from tqdm import tqdm
import pandas as pd
import os
from utils import tau, smooth


class Dataloader(object):

    def __init__(self):
        self.flux = [] # flux
        self.error = [] # error vector
        self.z = [] # z_qso
        self.redshift = [] # absorption redhsift
        self.mask = [] # mask array
        self.batch_size = None
        self.data_size = None
        self.cur = 0
        self.tau = tau
        self.Nb = None
        self.Nr = None
        self.device = None
        self.validation_dir = None

    def read_data(self, file):
        file = np.load(os.path.join(self.dir, file))
        flux = file['flux']
        error = file['error']
        mask = (file['flux']!=-999.)&(file['error']!=-999.)
        z = file['z']
        return flux, error, mask, z

    def read_validation(self, file):
        file = np.load(os.path.join(self.validation_dir, file))
        flux = file['flux']
        error = file['error']
        mask = (file['flux']!=-999.)&(file['error']!=-999.)
        z = file['z']
        return flux, error, mask, z

    def init(self, files, batch_size, device, dir, validation_dir = None, validation_files = None):
        """init this dataloader

        Args:
            files (list(str)): npz files to be read
            batch_size (int): batch size for this dataloader
            device (torch.device): device to run the code
            dir (dir): npz file dir
        """
        self.device = device
        self.dir = dir
        self.validation_dir = validation_dir
        loglam_min, loglam_max = np.log10(1030.), np.log10(1600.)
        loglam = np.arange(loglam_min, loglam_max, 1e-4)
        wav = 10**loglam
        self.Nb = np.sum(wav<1215.67)
        self.Nr = len(wav) - self.Nb
        with multiprocessing.Pool(24) as p:
            result = p.map(self.read_data, files)
        for r in tqdm(result):
            flux, error, mask, z = r
            self.flux.append(flux)
            self.error.append(error)
            self.z.append(z)
            self.mask.append(mask)
        if validation_dir is not None and validation_files is not None:
            with multiprocessing.Pool(24) as p:
                result = p.map(self.read_validation, validation_files)
            for r in tqdm(result):
                flux, error, mask, z = r
                self.flux.append(flux)
                self.error.append(error)
                self.z.append(z)
                self.mask.append(mask)
        self.flux = np.array(self.flux)
        self.error = np.array(self.error)
        self.z = np.array(self.z)
        self.mask = np.array(self.mask)
        self.redshift = (self.z + 1).reshape(-1, 1)*wav[:self.Nb]/1215.67 - 1
        self.batch_size = batch_size
        self.data_size = self.flux.shape[0]
        s = np.hstack((np.exp(1*self.tau(self.redshift)), np.ones((self.data_size, self.Nr), dtype=float)))
        self.mu_ = np.sum(self.flux*s, axis=0)/np.sum(self.flux!=0., axis=0)
        self.mu_ = smooth(self.mu_, window_len=16)

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
        s = np.hstack((np.exp(-1*self.tau(self.redshift[start: end])), np.ones((-start+end, self.Nr), dtype=float)))
        return torch.tensor(self.flux[start: end]-self.mu_*s, dtype=torch.float32).to(self.device),\
            torch.tensor(self.error[start: end], dtype=torch.float32).to(self.device), torch.tensor(self.redshift[start: end], dtype=torch.float32).to(self.device), \
                torch.tensor(self.mask[start: end], dtype=bool).to(self.device)

    def sample(self):
        """
        draw sample from this dataloader

        Returns:
            delta, error, redshift, mask (torch.tensor): sampled data
        """
        sig = np.random.randint(0, self.data_size, size=(self.batch_size, ))
        s = np.hstack((np.exp(-1.*self.tau(self.redshift[sig])), np.ones((self.batch_size, self.Nr), dtype=float)))
        return torch.tensor(self.flux[sig]-self.mu_*s, dtype=torch.tensor32).to(self.device),\
            torch.tensor(self.error[sig], dtype=torch.float32).to(self.device), torch.tensor(self.redshift[sig], dtype=torch.float32).to(self.device), \
                torch.tensor(self.mask[sig], dtype=bool).to(self.device)

    def rewind(self):
        """
        shuffle all the data and reset the dataloader
        """
        idx = np.arange(self.data_size)
        np.random.shuffle(idx)
        self.cur = 0
        self.flux = self.flux[idx]
        self.error = self.error[idx]
        self.z = self.z[idx]
        self.redshift = self.redshift[idx]
        self.mask = self.mask[idx]
