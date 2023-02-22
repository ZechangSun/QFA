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


def tau(z: torch.Tensor, which: Optional[str]='becker') -> torch.Tensor:
    """
    mean optical depth function
    ---------------------------------------------------
    Args:
        z (torch.Tensor (shape=(N, ), dtype=torch.float32)): redshift array
        which (str): which measurement to use ["becker", 'fg', 'kamble']
    Returns:
        effective optical depth: (torch.Tensor (shape=(N, ), dtype=torch.float32))
    """
    if which == 'becker':
        return _tau_becker(z)
    elif which == 'fg':
        return _tau_fg(z)
    elif which == 'kamble':
        return _tau_kamble(z)
    else:
        raise NotImplementedError("currently available mean optical depth function: ['becker', 'fg', 'kamble']")


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