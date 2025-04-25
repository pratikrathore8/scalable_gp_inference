import torch
from torch.optim import Adam

from .data_processing.load_torch import LOADERS

# For data loading
DATA_NAMES = list(LOADERS.keys())
DATA_SPLIT_PROPORTION = 0.1
DATA_SPLIT_SHUFFLE = True
DATA_STANDARDIZE = True

# For GP training
GP_TRAIN_DTYPE = torch.float32
GP_TRAIN_MAX_ITERS = 100
GP_TRAIN_NUM_TRIALS = 10
GP_TRAIN_OPT = Adam
GP_TRAIN_OPT_PARAMS = {"lr": 1e-1}
GP_TRAIN_SAVE_DIR = "saved_gp_hparams"
GP_TRAIN_SAVE_FILE_NAME = "gphparams.pkl"
GP_TRAIN_SUBSAMPLE_SIZE = 10000
