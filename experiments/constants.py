from torch.optim import Adam

from experiments.data_processing.load_torch import LOADERS

# For data loading
DATA_NAMES = list(LOADERS.keys())
DATA_SPLIT_PROPORTION = 0.1
DATA_SPLIT_SHUFFLE = True
DATA_STANDARDIZE = True

# For GP training
GP_TRAIN_DTYPE = "float64"
GP_TRAIN_MAX_ITERS = 100
GP_TRAIN_NUM_TRIALS = 10
GP_TRAIN_OPT = Adam
GP_TRAIN_OPT_PARAMS = {"lr": 1e-1}
GP_TRAIN_SAVE_DIR = "saved_gp_hparams"
GP_TRAIN_SAVE_FILE_NAME = "gp_hparams.pkl"
GP_TRAIN_SUBSAMPLE_SIZE = 10000

# Experiment parameters
EXPERIMENT_DATA_KERNEL_MAP = {
    "acsincome": "rbf",
    "yolanda": "rbf",
    "malonaldehyde": "matern52",
    "benzene": "matern52",
    "taxi": "rbf",
    "3droad": "matern32",
    "song": "matern32",
    "houseelec": "matern32",
}
EXPERIMENT_SEEDS = [0, 1, 2, 3, 4]
EXPERIMENT_KERNELS = [
    "rbf",
    "matern12",
    "matern32",
    "matern52",
]

# Parameters for GP inference
GP_INFERENCE_NUM_POSTERIOR_SAMPLES_MAP = {
    "acsincome": 64,
    "yolanda": 64,
    "malonaldehyde": 64,
    "benzene": 64,
    # If we are using 2048 random features, we will run out memory,
    # so we only focus on the posterior mean for taxi
    "taxi": 0,
    "3droad": 64,
    "song": 64,
    "houseelec": 64,
}
GP_INFERENCE_NUM_RANDOM_FEATURES = 2048
GP_INFERENCE_USE_FULL_KERNEL = True

# Optimizer parameters for GP inference
OPT_TYPES = ["pcg", "sap", "sdd"]
OPT_ATOL = 1e-12  # So small that nothing terminates early
OPT_RTOL = 1e-12  # So small that nothing terminates early
OPT_RANK = 100
OPT_DAMPING = "adaptive"
OPT_SAP_PRECONDITIONERS = ["nystrom", "identity"]
OPT_PCG_PRECONDITIONERS = ["nystrom"]
OPT_SDD_MOMENTUM = 0.9
OPT_SDD_STEP_SIZES_UNSCALED = [1, 10, 100]
OPT_SDD_THETA_UNSCALED = 100
OPT_MAX_PASSES_MAP = {
    "acsincome": 20,
    "yolanda": 50,
    "malonaldehyde": 50,
    "benzene": 50,
    "taxi": 2,
    "3droad": 50,
    "song": 50,
    "houseelec": 20,
}
OPT_NUM_BLOCKS_MAP = {
    "acsincome": 100,
    "yolanda": 100,
    "malonaldehyde": 100,
    "benzene": 100,
    "taxi": 2000,
    "3droad": 100,
    "song": 100,
    "houseelec": 100,
}

# Logging parameters
LOGGING_USE_WANDB = True
LOGGING_WANDB_PROJECT_BASE_NAME = "gp_inference"
