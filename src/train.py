import argparse
from model import QFA
from dataloader import Dataloader
from optimizer import Adam, step_scheduler
import logging
import os
import numpy as np
import pandas as pd
import torch
from config import get_config


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, required=False, help='configuration file')
parser.add_argument("--catalog", type=str, required=False, help="csv file which records the meta info for each spectra, see the example 'catalog.csv'")
parser.add_argument("--train_num", type=int, required=False, help="number of quasar spectra for model training")
parser.add_argument("--validation_catalog", type=str, required=False, help="csv file which records the meta info for each spectra, see the example 'catalog.csv'")
parser.add_argument("--validation_num", type=int, required=False, help="number of quasar spectra for validation set")
parser.add_argument("--batch_size", type=int, required=False, help='batch size for model training')
parser.add_argument("--n_epochs", type=int, required=False, help='number of training epochs')
parser.add_argument("--Nh", type=int, required=False, help="number of hidden variables")
parser.add_argument("--learning_rate", type=float, required=False, help="model learning rate (suggestion: learning_rate should be lower than 1e-2)")
parser.add_argument("--gpu", type=int, required=False, help="specify the GPU number for model training")
parser.add_argument("--snr_min", type=float, required=False, help="the lowest signal-to-noise ratio in the training set")
parser.add_argument("--snr_max", type=float, required=False, help="the highest signal-to-noise ratio in the training set")
parser.add_argument("--z_min", type=float, required=False, help='the min quasar redshift in the training set')
parser.add_argument("--z_max", type=float, required=False, help='the max quasar redshift in the training set')
parser.add_argument("--num_mask", type=int, required=False, help='the max number of masked region for spectra in the training set')
parser.add_argument("--alpha", type=float, required=False, help="decay rate for a step learning rate scheduler")
parser.add_argument("--step", type=int, required=False, help='learning rate decay step')
parser.add_argument("--weight_decay", type=float, required=False, help='parameter for L2 regularization')
parser.add_argument("--output_dir", type=str, required=False, help='output dir for model training result')
parser.add_argument("--data_dir", type=str, required=False, help='input dir for the training data')
parser.add_argument("--validation_dir", type=str, required=False, help='input dir for the validation data')
parser.add_argument("--validation", type=bool, required=False, help='input dir for the training data')
args = parser.parse_args()


config = get_config(args)

"""
if __name__ == "__main__":
    # gpu settings
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    catalog = pd.read_csv(args.catalog)
    criteria = (catalog['snr']>=args.snr_min) & (catalog['snr']<=args.snr_max) & (catalog['z']<=args.z_max) & (catalog['z'] >= args.z_min) & (catalog['num_mask'] <= args.num_mask)
    files = catalog['file'][criteria].values
    files = np.random.choice(files, args.train_num, replace=(len(files) < args.train_num))
    args.validation = False
    if args.validation == True:
        validation_catalog = pd.read_csv(args.validation_catalog)
        validation_criteria = (validation_catalog['snr']>=args.snr_min) & (validation_catalog['snr']<=args.snr_max) & (validation_catalog['z']<=args.z_max) & (validation_catalog['z'] >= args.z_min) & (validation_catalog['num_mask'] <= args.num_mask)
        validation_files = validation_catalog['file'][validation_criteria].values
        validation_files = np.random.choice(validation_files, args.validation_num, replace=(len(validation_files) < args.validation_num))
    else:
        args.validation_dir = None
        validation_files = None
    dataloader = Dataloader()
    dataloader.init(files, args.batch_size, device, args.data_dir, validation_dir=args.validation_dir, validation_files=validation_files)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    pd.DataFrame({"files": files}).to_csv(os.path.join(args.output_dir, 'training_data.csv'), index=False)
    if args.validation == True:
        validation_catalog[validation_criteria].to_csv(os.path.join(args.output_dir, 'validation_data.csv'), index=False)

    # set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    print("training...")
    # training
    model = QFA(dataloader.Nb, dataloader.Nr, args.Nh, device=device)
    scheduler = step_scheduler(args.alpha, args.step)
    optimizer = Adam(params=model.parameters, learning_rate=args.learning_rate, device=device, scheduler=scheduler, weight_decay=args.weight_decay)
    model.random_init_func()
    model.train(optimizer, dataloader, args.n_epochs, args.output_dir, logger=logger)
"""