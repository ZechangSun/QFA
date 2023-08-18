import argparse
from functools import partial
from QFA.model import QFA
from QFA.dataloader import Dataloader
from QFA.utils import tau as taufunc
from QFA.optimizer import Adam, step_scheduler
import logging
import os
import numpy as np
import torch
import time
from tqdm import tqdm
from QFA.config import get_config


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, required=False, help='configuration file')
parser.add_argument("--catalog", type=str, required=False, help="csv file which records the meta info for each spectra, see the example 'catalog.csv'")
parser.add_argument("--type", type=str, required=False, help='which mode to use [train or predict]...')
parser.add_argument("--data_num", type=int, required=False, help="number of quasar spectra for model training")
parser.add_argument("--validation_catalog", type=str, required=False, help="csv file which records the meta info for each spectra, see the example 'catalog.csv'")
parser.add_argument("--validation_num", type=int, required=False, help="number of quasar spectra for validation set")
parser.add_argument("--batch_size", type=int, required=False, help='batch size for model training')
parser.add_argument("--n_epochs", type=int, required=False, help='number of training epochs')
parser.add_argument("--Nh", type=int, required=False, help="number of hidden variables")
parser.add_argument("--tau", type=str, required=False, help='mean optical depth function')
parser.add_argument("--learning_rate", type=float, required=False, help="model learning rate (suggestion: learning_rate should be lower than 1e-2)")
parser.add_argument("--gpu", type=int, required=False, help="specify the GPU number for model training")
parser.add_argument("--snr_min", type=float, required=False, help="the lowest signal-to-noise ratio in the training set")
parser.add_argument("--snr_max", type=float, required=False, help="the highest signal-to-noise ratio in the training set")
parser.add_argument("--z_min", type=float, required=False, help='the min quasar redshift in the training set')
parser.add_argument("--z_max", type=float, required=False, help='the max quasar redshift in the training set')
parser.add_argument("--num_mask", type=int, required=False, help='the max number of masked region for spectra in the training set')
parser.add_argument("--decay_alpha", type=float, required=False, help="decay rate for a step learning rate scheduler")
parser.add_argument("--decay_step", type=int, required=False, help='learning rate decay step')
parser.add_argument("--weight_decay", type=float, required=False, help='parameter for L2 regularization')
parser.add_argument("--output_dir", type=str, required=False, help='output dir for model training result')
parser.add_argument("--data_dir", type=str, required=False, help='input dir for the training data')
parser.add_argument("--validation_dir", type=str, required=False, help='input dir for the validation data')
parser.add_argument("--validation", type=bool, required=False, help='input dir for the training data')
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs")
args = parser.parse_args()

config = get_config(args)


if __name__ == "__main__":
    # save config
    if not os.path.exists(config.DATA.OUTPUT_DIR):
        os.mkdir(config.DATA.OUTPUT_DIR)
    
    with open(os.path.join(config.DATA.OUTPUT_DIR, 'config.yaml'), 'w') as f:
        f.write(config.dump())
    
    # gpu settings
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert config.TYPE in ['train', 'predict'], "TYPE must be in ['train', 'predict']!"

    #load data
    dataloader = Dataloader(config)
    dataloader.set_device(device)

    if config.TYPE == 'train':
        # set up logger
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(os.path.join(config.DATA.OUTPUT_DIR, "log.txt"))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        print("training...")

        # training
        model = QFA(dataloader.Nb, dataloader.Nr, config.MODEL.NH, device=device, tau=partial(taufunc, which=config.MODEL.TAU))
        if os.path.exists(config.MODEL.RESUME):
            print(f"=> Resume from {config.MODEL.RESUME}")
            model.load_from_npz(config.MODEL.RESUME)
        scheduler = step_scheduler(config.TRAIN.DECAY_ALPHA, config.TRAIN.DECAY_STEP)
        optimizer = Adam(params=model.parameters, learning_rate=config.TRAIN.LEARNING_RATE, device=device, scheduler=scheduler, weight_decay=config.TRAIN.WEIGHT_DECAY)
        model.random_init_func()
        model.train(optimizer, dataloader, config.TRAIN.NEPOCHS, config.DATA.OUTPUT_DIR, logger=logger)
    elif config.TYPE == 'predict':
        print(f"try to predict {len(dataloader)} spectra...")
        model = QFA(dataloader.Nb, dataloader.Nr, config.MODEL.NH, device=device, tau=partial(taufunc, which=config.MODEL.TAU))
        print(f'=> Resume from {config.MODEL.RESUME}')
        model.load_from_npz(config.MODEL.RESUME)
        ts = time.time()
        output_dir = os.path.join(config.DATA.OUTPUT_DIR, 'predict')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for (f, e, z, m, p) in tqdm(dataloader):
            ll, hmean, hcov, cont, uncertainty = model.prediction_for_single_spectra(f, e, z, m)
            result_dict = {key: eval(f'{key}.cpu().detach().numpy()') for key in ['ll', 'hmean', 'hcov', 'cont', 'uncertainty']}
            name = os.path.basename(p)
            np.savez(os.path.join(output_dir, name), **result_dict)
        te = time.time()
        print(f'Finish predicting {len(dataloader)} spectra in {te-ts} seconds...')
    else:
        raise NotImplementedError(f"Mode {config.TYPE} hasn't been implemented!")
