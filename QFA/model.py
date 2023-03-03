"""
QuasarFactorAnalysis/model.py - generatively modeling quasar spectra && inferring the posterior distribution of quasar continua
Copyright 2022: Zechang Sun [https://zechangsun.github.io]
Email: szc22@mails.tsinghua.edu.cn
Reference: An Unsupervised Learning Approach for Quasar Continuum Prediction [https://arxiv.org/abs/2207.02788]
"""

from typing import Callable, Dict
import torch
import time
import os
import numpy as np
from .utils import MatrixInverse, MatrixLogDet, tauHI, omega_func
from .utils import tau as taufunc
from functools import partial

from torch.nn import functional as F


log2pi = 1.8378770664093453
default_tau = partial(taufunc, which='becker')


class QFA(object):

    def __init__(self, Nb: int, Nr: int, Nh: int,  device: torch.device, tau: Callable[[torch.Tensor, ], torch.Tensor]=default_tau, model_params:Dict[str, np.ndarray]=None) -> None:
        """
        Initialization for the spectra model
        Args:
            Nb (int): The number of pixels for the blue-side spectra
            Nr (int): The number of pixels for the red-side spectra
            Nh (int): The number of hidden variables
            device (torch.device): the device of the model
            tau (function): effective optical depth
            model_params (_type_, optional): dict {"F", "Psi", "omega", "tau0", "c0", "beta"} Defaults to None.
        """
        self.Nb = Nb
        self.Nr = Nr
        self.Nh = Nh
        self.device = device
        self.Npix = self.Nb + self.Nr # total number pixel number
        self.Nparams = self.Npix*self.Nh + self.Npix + self.Nb + 3 # total number of model parameters
        self.tau = tau
        self.min_value = 1e-3 # minimum tolerated value for omega & Psi
        self.max_value = 2. # maximum tolerated value for omega & Psi
        if model_params is not None:
            self.F = torch.tensor(model_params['F']).to(self.device)
            self.Psi = torch.tensor(model_params['Psi']).to(self.device)
            self.omega = torch.tensor(model_params['omega']).to(self.device)
            self.tau0 = torch.tensor(model_params['tau0']).to(self.device)
            self.c0 = torch.tensor(model_params['c0']).to(self.device)
            self.beta = torch.tensor(model_params['beta']).to(self.device)
        else:
            self.random_init_func()
        self.mu = None

    def random_init_func(self) -> None:
        """
        Random initialization model parameters with following stragetry:
        F: drawn from uniform(-0.5, 0.5)^(Npix x Nh)
        omega: constant 1.
        Psi: constant 1.
        tau0: constant 0.01
        c0: constant 0.1
        beta: constant 1.
        """
        self.F = torch.rand((self.Npix, self.Nh), dtype=torch.float32).to(self.device) - 0.5
        self.Psi = torch.ones((self.Npix, ), dtype=torch.float32).to(self.device)
        self.omega = torch.ones((self.Nb, ), dtype=torch.float32).to(self.device)
        self.tau0 = torch.tensor(0.02, dtype=torch.float32).to(self.device)
        self.c0 = torch.tensor(0.3, dtype=torch.float32).to(self.device)
        self.beta = torch.tensor(2., dtype=torch.float32).to(self.device)

    def forward(self, delta: torch.Tensor, error: torch.Tensor, zabs: torch.Tensor, mask: torch.Tensor):
        """Compute batch loss and gradient

        Args:
            delta (torch.tensor((batch_size, Npix), dtype=torch.float32)): delta fields for quasar spectra, mathemmatically, \delta = S - A@\mu
            error (torch.tensor((batch_size, Npix), dtype=torch.float32)): error vector for quasar spectra
            zabs (torch.tensor((batch_size, Nb), dtype=torch.float32)): zabs array for Ly$\alpha$ absorption systems
            mask (torch.tensor((batch_size, Npix), dtype=torch.float32)): mask array which indicates the available part of the spectra

        Returns:
            batch loglikelihood (torch.tensor((1, ), dtype=torch.float32)): batch averaged loglikelihood
            batch gradient (dict): batch averaged gradient for each parameter
        """
        batch_size = delta.shape[0]
        loss = 0.
        gradient = {
            "F": torch.zeros_like(self.F, dtype=torch.float32).to(self.device),
            "Psi": torch.zeros_like(self.Psi, dtype=torch.float32).to(self.device),
            "omega": torch.zeros_like(self.omega, dtype=torch.float32).to(self.device),
            "tau0": torch.zeros_like(self.tau0, dtype=torch.float32).to(self.device),
            "c0": torch.zeros_like(self.c0, dtype=torch.float32).to(self.device),
            "beta": torch.zeros_like(self.beta, dtype=torch.float32).to(self.device)
        }
        non_zero_values = {key: torch.zeros_like(gradient[key], dtype=torch.float32).to(self.device) for key in gradient}
        for d, e, r, m in zip(delta, error, zabs, mask):
            single_loss, single_gradient = self.loglikelihood_and_gradient_for_single_spectra(d, e, r, m)
            loss += single_loss/batch_size
            for key in gradient:
                gradient[key] += single_gradient[key]
                non_zero_values[key] += (single_gradient[key]!=0.)
        gradient = {key: gradient[key]/non_zero_values[key] for key in gradient}
        return loss, gradient

    def loglikelihood_and_gradient_for_single_spectra(self, delta: torch.Tensor, error: torch.Tensor, zabs: torch.Tensor, mask: torch.Tensor):
        """
        Compute the nagetive loglikelihood and the gradient for a single spectra

        Args:
            delta (torch.tensor((Npix, ), dtype=torch.float32)): delta fields for quasar spectra, mathemmatically, \delta = S - A@\mu
            error (torch.tensor((Npix, ), dtype=torch.float32)): error vector for quasar spectra
            zabs (torch.tensor((Nb, ), dtype=torch.float32)): zabs array for Ly$\alpha$ absorption systems
            mask (torch.tensor((Npix, ), dtype=torch.float32)): mask array which indicates the available part of the spectra

        Returns:
            loglikelihood (torch.tensor((1, ), dtype=torch.float32)): nagetive loglikelihood for this spectra
            gradient (dict): gradient for each parameter
        """
        blue_mask, red_mask = mask[:self.Nb], mask[self.Nb:]
        Nb, Nr = torch.sum(blue_mask), torch.sum(red_mask)
        Npix = Nb + Nr
        masked_delta, masked_error, masked_zabs = delta[mask], error[mask], zabs[blue_mask]
        A = torch.hstack((torch.exp(-1.*self.tau(masked_zabs)), torch.ones(Nr, dtype=torch.float32).to(self.device)))
        diagA = torch.diag(A)
        F = diagA@self.F[mask, :]
        Psi = A*self.Psi[mask]*A
        zdep = omega_func(masked_zabs, self.tau0, self.beta, self.c0)
        omega = torch.hstack((self.omega[blue_mask]*zdep, torch.zeros(Nr, dtype=torch.float32).to(self.device)))
        diag = Psi + omega + masked_error*masked_error
        invSigma = MatrixInverse(F, diag, self.device)
        logDet = MatrixLogDet(F, diag, self.device)
        masked_delta = masked_delta[:, None]
        loglikelihood = 0.5*(masked_delta.mT @ invSigma @ masked_delta + Npix * log2pi + logDet)
        partialSigma = 0.5*(invSigma-invSigma@masked_delta@masked_delta.mT@invSigma)
        partialF = 2*diagA@partialSigma@diagA@F
        diagPartialSigma = torch.diag(partialSigma)
        partialPsi = A*diagPartialSigma*A
        partialOmega = diagPartialSigma[:Nb]*zdep
        root = 1. - tauHI(masked_zabs, self.tau0, self.beta) - self.c0
        partialTau0 = -1.*torch.sum(diagPartialSigma[:Nb]*omega[:Nb]*zdep*2.*root*(torch.pow(1.+masked_zabs, self.beta)))
        partialBeta = -1.*torch.sum(diagPartialSigma[:Nb]*omega[:Nb]*zdep*2.*root*(self.tau0*torch.pow((1+masked_zabs), self.beta)*torch.log((1+masked_zabs))))
        partialC0 = -1.*torch.sum(diagPartialSigma[:Nb]*omega[:Nb]*zdep*2.*root)
        gradientF = torch.zeros((self.Npix, self.Nh), dtype=torch.float32).to(self.device)
        gradientF[mask, :] = partialF
        gradientOmega = torch.zeros((self.Nb, ), dtype=torch.float32).to(self.device)
        gradientOmega[blue_mask] = partialOmega
        gradientPsi = torch.zeros((self.Npix, ), dtype=torch.float32).to(self.device)
        gradientPsi[mask] = partialPsi
        return loglikelihood, {
            "F": gradientF,
            "Psi": gradientPsi,
            "omega": gradientOmega,
            "tau0": partialTau0,
            "c0": partialC0,
            "beta": partialBeta
        }

    def prediction_for_single_spectra(self, flux: torch.Tensor, error: torch.Tensor, zabs: torch.Tensor, mask: torch.Tensor):
        blue_mask, red_mask = mask[:self.Nb], mask[self.Nb:]
        Nb, Nr = torch.sum(blue_mask), torch.sum(red_mask)
        Npix = Nb + Nr
        masked_flux, masked_error, masked_zabs = flux[mask], error[mask], zabs[blue_mask]
        A = torch.hstack((torch.exp(-1.*self.tau(masked_zabs)), torch.ones(Nr, dtype=torch.float32).to(self.device)))
        masked_delta = masked_flux - self.mu[mask]*A
        diagA = torch.diag(A)
        F = diagA@self.F[mask, :]
        Psi = A*self.Psi[mask]*A
        zdep = omega_func(masked_zabs, self.tau0, self.beta, self.c0)
        omega = torch.hstack((self.omega[blue_mask]*zdep, torch.zeros(Nr, dtype=torch.float32).to(self.device)))
        diag = Psi + omega + masked_error*masked_error
        invSigma = MatrixInverse(F, diag, self.device)
        logDet = MatrixLogDet(F, diag, self.device)
        masked_delta = masked_delta[:, None]
        loglikelihood = 0.5*(masked_delta.mT @ invSigma @ masked_delta + Npix * log2pi + logDet)
        Sigma_e = torch.diag(1./diag)
        hcov = torch.linalg.inv(torch.eye(self.Nh, dtype=torch.float).to(self.device) + F.T@Sigma_e@F)
        hmean = hcov@F.T@Sigma_e@masked_delta
        return loglikelihood, hmean, hcov, (self.F@hmean).squeeze() + self.mu, torch.diag(self.F@hcov@self.F.T)**0.5


    def train(self, optimizer, dataloader, n_epochs, output_dir="./result", save_interval=5, smooth_interval=5, quiet=False, logger=None):
        """ model training given data and optimizer

        Args:
            optimizer (optimizers.optimizer (see optmizers.py)): self-defined optimizer class
            dataloader (dataloader.Dataloader (see dataloader.py)): dataloader.class
            n_epochs (int): number of training epochs
            output_dir (str, optional): outpur dir of training result if not exists then create one. Defaults to "./result".
            save_interval (int, optional): save the parameters after save_interval's epochs. Defaults to 10.
            smooth_interval (int, optional): smooth the parameter after smooth_interval's epochs. Defaults to 5.
            quiet (bool, optional): whether print training information to terminal. Defaults to False.
            logger (logging.logger, optional): used to record training information if None, then record nothing. Defaults to None.

        Returns:
            None
        """
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_dir = os.path.join(output_dir, 'checkpoints')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.mu = torch.tensor(dataloader.mu, dtype=torch.float32).to(self.device)
        Niter = dataloader.data_size//dataloader.batch_size
        def step(i):
            dataloader.rewind()
            total_loss = 0.
            start_time = time.time()
            while dataloader.have_next_batch():
                d, e, z, m = dataloader.next_batch()
                loss, grads = self.forward(d, e, z, m)
                total_loss += loss.item()/Niter
                self.parameters = optimizer.update(self.parameters, grads)
            optimizer.step()
            end_time = time.time()
            if not quiet:
                print("epoch: {:03d}/{:03d}  ;  loss:  {:.2f}  ;  time:  {:.2f} s ".format(i, n_epochs, total_loss, end_time-start_time))
            if logger is not None:
                logger.info("epoch: {:03d}/{:03d}  ;  loss:  {:.2f}  ;  time:  {:.2f} s ".format(i, n_epochs, total_loss, end_time-start_time))
            return total_loss
        for epoch in range(n_epochs):
            loss = step(epoch)
            if loss < 0.:
                self.smooth()
                self.save_to_npz(output_dir, 'model_parameters_epoch_%02i.npz'%(epoch+1))
                break
            if (epoch + 1) % smooth_interval == 0:
                self.smooth()
            if (epoch + 1) % save_interval == 0:
                self.save_to_npz(output_dir, 'model_parameters_epoch_%02i.npz'%(epoch+1))

    def clip(self):
        """
        Make Sure each parameter which may lead to numerical problem to stay in a appropriate interval after each update
        """
        self.omega = torch.clip(self.omega, min=self.min_value, max=self.max_value)
        self.Psi = torch.clip(self.Psi, min=self.min_value, max=self.max_value)
        self.tau0 = torch.clip(self.tau0, min=0., max=1.)
        self.beta = torch.clip(self.beta, min=0.1, max=5.)
        self.c0 = torch.clip(self.c0, min=-5., max=5.)

    def smooth(self):
        """
        Smooth omega, Psi, F along the Npix axis
        """
        omega = self.omega.reshape(1, -1)
        self.omega = F.avg_pool1d(omega, kernel_size=15, stride=1, padding=7, count_include_pad=False).squeeze()
        Psi = self.Psi.reshape(1, -1)
        self.Psi = F.avg_pool1d(Psi, kernel_size=15, stride=1, padding=7, count_include_pad=False).squeeze()
        F_ = self.F.data.reshape(1, self.Npix, self.Nh)
        self.F = F.avg_pool2d(F_, kernel_size=(31, 1), stride=(1, 1), padding=(15, 0), count_include_pad=False).squeeze()

    def save_to_npz(self, output_dir: str, file_name: str):
        """
        Save model parameters to a npz file with format like:

                    mu: model.mu (np.ndarray)
                    F : model.F  (np.ndarray)
                    omega: model.omega (np.ndarray)
       npz file --> tau0: model.tau0 (np.ndarray)
                    c0: model.c0 (np.ndarray)
                    beta: model.beta (np.ndarray)
        -----------------------------------------------------
        Args:
            output_dir (str): dir for the npz file, if not exists, then create one
            file_name (str): file name for the npz file

        ----------------------------------------------------
        Returns:
            None
        """
        mu = self.mu.cpu().detach().numpy()
        F_ = self.F.cpu().detach().numpy()
        Psi = self.Psi.cpu().detach().numpy()
        omega = self.omega.cpu().detach().numpy()
        tau0, c0, beta = self.tau0.cpu().detach().numpy(), self.c0.cpu().detach().numpy(), self.beta.cpu().detach().numpy()
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        np.savez(os.path.join(output_dir, file_name), mu=mu, F=F_, Psi=Psi, omega=omega, tau0=tau0, c0=c0, beta=beta)

    def load_from_npz(self, path: str):
        """Load model parameter from npz file

        Args:
            path (str): npz file for model parameters
        """
        file = np.load(path)
        self.mu = torch.tensor(file['mu'], dtype=torch.float32, device=self.device)
        self.F = torch.tensor(file['F'], dtype=torch.float32, device=self.device)
        self.omega = torch.tensor(file['omega'], dtype=torch.float32, device=self.device)
        self.Psi = torch.tensor(file['Psi'], dtype=torch.float32, device=self.device)
        self.tau0 = torch.tensor(file['tau0'], dtype=torch.float32, device=self.device)
        self.beta = torch.tensor(file['beta'], dtype=torch.float32, device=self.device)
        self.c0 = torch.tensor(file['beta'], dtype=torch.float32, device=self.device)

    @property
    def parameters(self):
        return {
            "F": self.F,
            "Psi": self.Psi,
            "omega": self.omega,
            "tau0": self.tau0,
            "c0": self.c0,
            "beta": self.beta
        }

    @parameters.setter
    def parameters(self, params_dict):
        self.F = params_dict['F']
        self.Psi = params_dict['Psi']
        self.omega = params_dict['omega']
        self.tau0 = params_dict['tau0']
        self.c0 = params_dict['c0']
        self.beta = params_dict['beta']
        self.clip()
