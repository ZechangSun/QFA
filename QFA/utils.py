"""
QuasarFactorAnalysis/utils.py - useful functions for model building
Copyright 2022: Zechang Sun [https://zechangsun.github.io]
Email: szc22@mails.tsinghua.edu.cn
Reference: An Unsupervised Learning Approach for Quasar Continuum Prediction [https://arxiv.org/abs/2207.02788]
"""
import torch
import numpy as np
from typing import Optional


def MatrixInverse(M: torch.Tensor, D: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Efficient and stable matrix inverse function for matrix with structure:
    $$
        Matrix = MM^T + D
    $$
    Here M is a Npix x Nh matrix and D is a Npix x Npix diagonal matrix, usually Npix >> Nh
    ---------------------------------------------------------------
    Args:
        M (torch.Tensor (shape=(Npix, Nh), dtype=torch.float32)): a Npix x Nh matrix
        D (torch.Tensor (shape=(Npix, ), dtype=torch.float32)): the diagonal elements for D
        device (torch.device): specify which device to be used
    Returns:
        inverse matrix of MM^T + D
        shape: (Npix, Npix)
        dtype: torch.float32
    """
    _, Nh = M.shape
    diagD = torch.diag(1./D)
    I = torch.eye(Nh, dtype=torch.float32).to(device)
    return diagD - diagD @ M @ torch.linalg.inv(I + M.T @ diagD @ M) @ M.T @ diagD


def MatrixLogDet(M: torch.Tensor, D: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Efficient and stable matrix log determinant function for matrix with structure:
    $$
        Matrix = MM^T + D
    $$
    Here M is a Npix x Nh matrix and D is a Npix x Npix diagonal matrix, usually Npix >> Nh
    ---------------------------------------------------------------
    Args:
        M (torch.array (shape=(Npix, Nh), dtype=torch.float32)): a Npix x Nh matrix
        D (torch.array (shape=(Npix,   ), dtype=torch.float32)): the diagonal elements for D
        device (torch.device): specify which device to be used
    Returns:
        Log determinant for MM^T + D
        type: torch.float32
    """
    _, Nh = M.shape
    diagD = torch.diag(1./D)
    I = torch.eye(Nh, dtype=torch.float32).to(device)
    return torch.sum(torch.log(D)) + torch.log(torch.linalg.det(I + M.T @ diagD @ M))


def tauHI(z: torch.Tensor, tau0: torch.float32, beta: torch.float32) -> torch.Tensor:
    """
    Simple power law for effective optical depth of absorption systems
    $$
        tau(z) = tau0*(1+z)**beta
    $$
    -------------------------------------------------
    Args:
        z (torch.Tensor (shape=(N, ), dtype=torch.float32)): redshift array
        tau0 (torch.float32): power law amplitude
        beta (torch.float32): power law index

    Returns:
        effective optical depth (torch.Tensor (shape=(N, ), dtype=torch.float32))
    """
    return tau0 * torch.pow((1. + z), beta)


def omega_func(z: torch.Tensor, tau0: torch.float32, beta: torch.float32, c0: torch.float32) -> torch.Tensor:
    """
    Absorption noise evolution function
    $$
        omega(z) = (1-tau0*(1+z)^beta-c0)**2
    $$
    ---------------------------------------------
    Args:
        z (torch.Tensor (shape=(N, ), dtype=torch.float32)): redshift array
        tau0 (torch.float32): effective optical depth parameter
        beta (torch.float32): effective optical depth parameter
        c0 (torch.float32): bias term

    Returns:
        omega (torch.Tensor (shape=(N, ), dtype=torch.float32)) : absorption noise redshift evolution
    """
    root = 1. - c0 - torch.exp(-1.*tauHI(z, tau0, beta))
    return root*root


def _tau_becker(z: torch.Tensor)->torch.Tensor:
    """
    mean optical depth measured by Becker et al. 2012 [https://arxiv.org/abs/1208.2584]
    ----------------------------------------------------------------------
    Args:
        z (torch.Tensor (shape=(N, ), dtype=torch.float32)): redshift array

    Returns:
        effective optical depth: (torch.Tensor (shape=(N, ), dtype=torch.float32))
    """
    tau0, beta, C, z0 = (0.751, 2.90, -0.132, 3.5)
    return tau0 * ((1+z)/(1+z0)) ** beta + C


def _tau_fg(z: torch.Tensor)->torch.Tensor:
    """
    mean optical depth measured by Faucher Giguere et al. 2008 [https://iopscience.iop.org/article/10.1086/588648]
    -------------------------------------------------------------------------
    Args:
        z (torch.Tensor (shape=(N, ), dtype=torch.float32)): redshift array
    
    Returns:
        effective optical depth: (torch.Tensor (shape=(N, ), dtype=torch.float32))
    """
    tau0, beta = 0.0018, 3.92
    return tau0 * (1+z) ** beta


def _tau_kamble(z: torch.Tensor)->torch.Tensor:
    """
    mean optical depth measured by Kamble et al. 2020 [https://ui.adsabs.harvard.edu/abs/2020ApJ...892...70K/abstract]
    ------------------------------------------------------------------------------------------
    Args:
        z (torch.Tensor (shape=(N, ), dtype=torch.float32)): redshift array
    
    Returns:
        effective optical depth: (torch.Tensor (shape=(N, ), dtype=torch.float32))
    """
    tau0, beta = 5.54*1e-3, 3.182
    return tau0 * (1+z) ** beta


def _tau_mock(z: torch.Tensor)->torch.Tensor:
    """
    mean optical depth from mock literature, Bautista et al. 2015 [https://iopscience.iop.org/article/10.1088/1475-7516/2015/05/060]
    """
    return 0.2231435513142097*((1+z)/3.25)**3.2

# calculate the coefficients of optical depth for each Lyman series line [https://arxiv.org/abs/2003.11036 Eq17]
lyseries = np.genfromtxt('./Lyman_series.csv', delimiter=',', names=True, 
                         dtype=[('name', 'U10'), ('f', 'f8'), ('lambda', 'f8'), ('coeff', 'f8')])
lya_coeff = lyseries[0]['lambda'] * lyseries[0]['f']
lyseries['coeff'] = lyseries['lambda'] * lyseries['f'] / lya_coeff

def tau(z: torch.Tensor, which: Optional[str]='becker', series: Optional[int]=1) -> torch.Tensor:
    """
    mean optical depth function
    ---------------------------------------------------
    Args:
        z (torch.Tensor (shape=(N, ), dtype=torch.float32)): redshift array
        which (str): which measurement to use ["becker", 'fg', 'kamble']
        series (int): the Lyman series line to be calculated. e.g. 1=alpha, 2=beta, 3=gamma, ... (up to 30)
    Returns:
        effective optical depth: (torch.Tensor (shape=(N, ), dtype=torch.float32))
    """
    coeff = lyseries[series-1]['coeff']

    if which == 'becker':
        return _tau_becker(z) * coeff
    elif which == 'fg':
        return _tau_fg(z) * coeff
    elif which == 'kamble':
        return _tau_kamble(z) * coeff
    elif which == 'mock':
        return _tau_mock(z) * coeff
    else:
        raise NotImplementedError("currently available mean optical depth function: ['becker', 'fg', 'kamble']")


def tau_total(wav_grid: torch.Tensor, zqso: torch.Tensor, which: Optional[str]='becker') -> torch.Tensor:
    """
    total optical depth function
    ---------------------------------------------------
    Args:
        wav_grid (torch.Tensor (shape=(N, ), dtype=torch.float32)): wavelength grid
        zqso (torch.Tensor (shape=(N, ), dtype=torch.float32)): QSO redshift array
        which (str): which measurement to use ["becker", 'fg', 'kamble']
    Returns:
        effective optical depth: (torch.Tensor (shape=(N, ), dtype=torch.float32))
    """
    # calculate the highest level of Lyman series that needs to be considered
    wav_start = wav_grid[0]
    ly_level = 0
    while wav_start < lyseries[ly_level]['lambda']:
        ly_level += 1
        if ly_level == len(lyseries):
            break
    if ly_level == 0:
        raise ValueError("Wavelength grid does not cover Lyman series lines")

    # calculate the total optical depth
    Nb = np.sum(wav_grid<lyseries[0]['lambda'])
    taus = np.zeros_like(zqso).reshape(-1, 1) * np.zeros_like(wav_grid[:Nb])
    for i in range(ly_level):
        Nb_this = np.sum(wav_grid<lyseries[i]['lambda'])
        zabs_this = (zqso + 1).reshape(-1, 1) * wav_grid[:Nb_this]/lyseries[i]['lambda'] - 1
        taus[:, 0:Nb_this] += tau(zabs_this, which=which, series=i+1)
    
    return taus


def smooth(s: np.ndarray, window_len: Optional[int]=32):
    """Smooth curve s with corresponding window length

    Args:
        s (numpy.ndarray (shape: (N, ), dtype=float)): a 1d curve
        window_len (int, optional): smoothing window. Defaults to 32.

    Returns:
        smoothed curve (numpy.ndarray (shape: (N, ), dtype=float))
    """
    s = np.r_[s[window_len-1:0:-1], s, s[-2:-window_len-1:-1]]
    kernel = np.ones(window_len, dtype=float)/window_len
    y = np.convolve(kernel, s, mode='valid')
    return y[int(window_len/2-1):-int(window_len/2)]