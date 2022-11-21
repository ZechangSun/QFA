"""
QuasarFactorAnalysis/utils.py - optimizers
Copyright 2022: Zechang Sun [https://zechangsun.github.io]
Email: szc22@mails.tsinghua.edu.cn
Reference: An Unsupervised Learning Approach for Quasar Continuum Prediction [https://arxiv.org/abs/2207.02788]
"""
import torch
from typing import Callable, Dict


class Adam(object):

    def __init__(self, params: Dict[str, torch.Tensor], device: torch.device, scheduler=Callable[[float, int], Callable[[int, float], float]], learning_rate: float=1e-2, b1: float=0.9, b2: float=0.999, eps: float=1e-8, weight_decay: float=1e-3) -> None:
        """Adam optimizer

        Args:
            params (dict{str: torch.tensor}): model trainable parameters
            device (torch.device): specify which GPU to use
            scheduler (function(i, learning_rate), optional): learning rate decay scheduler. Defaults to None.
            learning_rate (float, optional): learning rate. Defaults to 1e-2.
            b1 (float, optional): Adam parameter. Defaults to 0.9.
            b2 (float, optional): Adam parameter. Defaults to 0.999.
            eps (float, optional): Adam parameter. Defaults to 1e-8.
            weight_decay (float, optional): L2 norm parameter. Defaults to 1e-3.
        """
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.device = device
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.m = {v: torch.zeros_like(params[v], dtype=torch.float32).to(self.device) for v in params}
        self.v = {v: torch.zeros_like(params[v], dtype=torch.float32).to(self.device) for v in params}
        self.i = 0

    def update(self, params, g):
        """update parameters based on optimizer state and gradients

        Args:
            params (dict{str: torch.tensor}): model parameters
            g (dict{str: torch.tensor}): model gradients

        Returns:
            updated parameters (dict{str: torch.tensor}): updated model parameters
        """
        g = {v: g[v]+self.weight_decay*params[v] for v in g}
        self.m = {v: (1-self.b1)*g[v]+self.b1*self.m[v] for v in g}
        self.v = {k: (1-self.b2)*g[k]*g[k]+self.b2*self.v[k] for k in g}
        mhat = {v: self.m[v]/(1.-self.b1**(self.i+1)) for v in g}
        vhat = {k: self.v[k]/(1.-self.b2**(self.i+1)) for k in g}
        return {v: params[v]-self.scheduled_lr*mhat[v]/(torch.sqrt(vhat[v])+self.eps) for v in params}

    def reset(self, params):
        """
        reset the optimizer

        Args:
            params (dict{str: torch.tensor}): model parameters
        """
        self.m = {v: torch.zeros_like(params[v], dtype=torch.float32).to(self.device) for v in params}
        self.v = {v: torch.zeros_like(params[v], dtype=torch.float32).to(self.device) for v in params}
        self.i = 0

    def step(self):
        """
        one step forward for the optmizer
        """
        self.i += 1

    @property
    def scheduled_lr(self):
        if callable(self.scheduler):
            return self.scheduler(self.i, self.learning_rate)
        else:
            return self.learning_rate


def step_scheduler(alpha: float, step: int) -> Callable[[int, float], float]:
    """
    step scheduler, update rule: lr = lr_0 * alpha**((1+i)/step)
    
    Args:
        alpha (float): decay parameter
        step (int): decay after step
    
    return:
        scheduler (callable): scheduler function in training
    """
    def scheduler(i, lr):
        """
        scheduler function

        Args:
            i (int): current iteration
            lr (float): learning rate
        """
        return lr*alpha**((i+1)//step)
    return scheduler