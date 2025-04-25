from torch.optim import Adam

# For data loading
DATA_SPLIT_PROPORTION = 0.1
DATA_STANDARDIZE = True

# For GP training
GP_TRAIN_MAX_ITERS = 100
GP_TRAIN_OPT = Adam
GP_TRAIN_OPT_PARAMS = {"lr": 1e-1}
