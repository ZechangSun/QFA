import argparse
from QFA.model import QFA
from QFA.dataloader import Dataloader
from QFA.optimizer import Adam, step_scheduler
import logging
import os
import torch
from QFA.config import get_config


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, required=False, help='configuration file')
parser.add_argument("--catalog", type=str, required=False, help="csv file which records the meta info for each spectra, see the example 'catalog.csv'")
parser.add_argument("--data_num", type=int, required=False, help="number of quasar spectra for model training")
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
    # gpu settings
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #load data
    dataloader = Dataloader(config)
    dataloader.set_device(device)

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
    model = QFA(dataloader.Nb, dataloader.Nr, config.MODEL.NH, device=device)
    if os.path.exists(config.MODEL.RESUME):
        print(f"=> Resume from {config.MODEL.RESUME}")
        model.load_from_npz(config.MODEL.RESUME)
    scheduler = step_scheduler(config.TRAIN.DECAY_ALPHA, config.TRAIN.DECAY_STEP)
    optimizer = Adam(params=model.parameters, learning_rate=config.TRAIN.LEARNING_RATE, device=device, scheduler=scheduler, weight_decay=config.TRAIN.WEIGHT_DECAY)
    model.random_init_func()
    model.train(optimizer, dataloader, config.TRAIN.NEPOCHS, config.DATA.OUTPUT_DIR, logger=logger)
