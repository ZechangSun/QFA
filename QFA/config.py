"""
config.py - load configuration for training
Copyright 2022: Zechang Sun
Email: szc22@mails.tsinghua.edu.cn
Url: https://zechangsun.github.io
"""

import os
import yaml
import numpy as np
from yacs.config import CfgNode as CN


_C = CN()

# Base config files
_C.BASE = ['']
_C.GPU = 0

#------------------
# Data settings
#------------------
_C.DATA = CN()
_C.DATA.DATA_DIR = ''
_C.DATA.VALIDATION_DIR = ''
_C.DATA.OUTPUT_DIR = ''
_C.DATA.CATALOG = ''
_C.DATA.VALIDATION_CATALOG = ''
_C.DATA.DATA_NUM = 10000
_C.DATA.VALIDATION_NUM = 1000
_C.DATA.BATCH_SIZE = 500
_C.DATA.SNR_MIN = 2
_C.DATA.SNR_MAX = 100
_C.DATA.Z_MIN = 2
_C.DATA.Z_MAX = 3.5
_C.DATA.NUM_MASK = 0
_C.DATA.LAMMIN = 1030.
_C.DATA.LAMMAX = 1600.
_C.DATA.LOGLAM_DELTA = 1e-4
_C.DATA.NPROCS = 24
_C.DATA.VALIDATION = False


#------------------
# Model settings
#------------------
_C.MODEL = CN()
_C.MODEL.NH = 8
_C.MODEL.TAU = 'becker'
_C.MODEL.RESUME = ''


#------------------
# Training settings
#------------------
_C.TRAIN = CN()
_C.TRAIN.NEPOCHS = 500
_C.TRAIN.LEARNING_RATE = 1e-3
_C.TRAIN.WEIGHT_DECAY = 1e-1
_C.TRAIN.DECAY_ALPHA = 0.9
_C.TRAIN.DECAY_STEP = 10
_C.TRAIN.WINDOW_LENGTH_FOR_MU = 16


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(config, os.path.join(os.path.dirname(cfg_file), cfg))
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    if isinstance(args.cfg, str):
        _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    
    def _check_args(name):
        if hasattr(args, name) and eval(f"args.{name}"):
            return True
        return False

    if _check_args('gpu'):
        config.GPU = args.gpu

    if _check_args('n_epochs'):
        config.TRAIN.NEPOCHS = args.n_epochs
    if _check_args('learning_rate'):
        config.TRAIN.LEARNING_RATE = args.learning_rate
    if _check_args('weight_decay'):
        config.TRAIN.WEIGHT_DECAY = args.weight_decay
    if _check_args('decay_alpha'):
        config.TRAIN.DECAY_ALPHA = args.decay_alpha
    if _check_args('decay_step'):
        config.TRAIN.DECAY_STEP = args.decay_step

    if _check_args('data_dir'):
        config.DATA.DATA_DIR = args.data_dir
    if _check_args('validation_dir'):
        config.DATA.VALIDATION_DIR = args.validation_dir
    if _check_args('output_dir'):
        config.DATA.OUTPUT_DIR = args.output_dir
    if _check_args('catalog'):
        config.DATA.CATALOG = args.catalog
    if _check_args('validation_catalog'):
        config.DATA.VALIDATION_CATALOG = args.validation_catalog
    if _check_args('data_num'):
        config.DATA.DATA_NUM = args.data_num
    if _check_args('validation_num'):
        config.DATA.VALIDATION_NUM = args.validation_num
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('snr_min'):
        config.DATA.SNR_MIN = args.snr_min
    if _check_args('snr_max'):
        config.DATA.SNR_MAX = args.snr_max
    if _check_args('z_min'):
        config.DATA.Z_MIN = args.z_min
    if _check_args('z_max'):
        config.DATA.Z_MAX = args.z_max
    if _check_args('num_mask'):
        config.DATA.NUM_MASK = args.num_mask
    if _check_args('nprocs'):
        config.DATA.NPROCS = args.nprocs
    if _check_args('validation'):
        config.DATA.VALIDATION = args.validation
    if _check_args('tau'):
        config.MODEL.TAU = args.tau


    config.freeze()


def get_config(args):
    config = _C.clone()
    update_config(config, args)
    
    return config