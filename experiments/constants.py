from torch.optim import Adam

from experiments.data_processing.load_torch import LOADERS

# For data loading
DATA_NAMES = list(LOADERS.keys())
DATA_SPLIT_PROPORTION_MAP = {
    "ESR2": 0.1,
    "F2": 0.1,
    "KIT": 0.1,
    "PARP1": 0.1,
    "PGR": 0.1,
    "acsincome": 0.1,
    "yolanda": 0.1,
    "malonaldehyde": 0.1,
    "benzene": 0.1,
    "taxi": 0.01,
    "3droad": 0.1,
    "song": 0.1,
    "houseelec": 0.1,
}
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
    "ESR2": "tanimoto",
    "F2": "tanimoto",
    "KIT": "tanimoto",
    "PARP1": "tanimoto",
    "PGR": "tanimoto",
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
EXPERIMENT_KERNELS = list(
    set(EXPERIMENT_DATA_KERNEL_MAP.values())
)  # Unique kernels from the dataset mapping

# Parameters for GP inference
GP_INFERENCE_NUM_POSTERIOR_SAMPLES_MAP = {
    "ESR2": 0,
    "F2": 0,
    "KIT": 0,
    "PARP1": 0,
    "PGR": 0,
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
# Don't use full kernel for residual computation for datasets
# with large number of samples or large number of features
GP_INFERENCE_USE_FULL_KERNEL_MAP = {
    "ESR2": False,
    "F2": False,
    "KIT": False,
    "PARP1": False,
    "PGR": False,
    "acsincome": False,
    "yolanda": True,
    "malonaldehyde": True,
    "benzene": True,
    "taxi": False,
    "3droad": True,
    "song": True,
    "houseelec": False,
}

# Optimizer parameters for GP inference
OPT_TYPES = ["sap", "sdd", "pcg"]
OPT_ATOL = 1e-16  # So small that nothing terminates early
OPT_RTOL = 1e-16  # So small that nothing terminates early
OPT_RANK = 100
OPT_DAMPING = "adaptive"
OPT_SAP_PRECONDITIONERS = ["nystrom", "identity"]
OPT_SAP_PRECISIONS = ["float32"]
OPT_PCG_PRECONDITIONERS = ["nystrom"]
OPT_PCG_PRECISIONS = ["float32"]  # ["float32", "float64"]
OPT_SDD_MOMENTUM = 0.9
OPT_SDD_STEP_SIZES_UNSCALED = [1, 10, 100]
OPT_SDD_THETA_UNSCALED = 100
OPT_SDD_PRECISIONS = ["float32"]
OPT_MAX_PASSES_MAP = {
    "ESR2": 50,
    "F2": 50,
    "KIT": 50,
    "PARP1": 50,
    "PGR": 50,
    "acsincome": 20,
    "yolanda": 50,
    "malonaldehyde": 50,
    "benzene": 50,
    "taxi": 1,
    "3droad": 50,
    "song": 50,
    "houseelec": 20,
}
OPT_NUM_BLOCKS_MAP = {
    "ESR2": 100,
    "F2": 100,
    "KIT": 100,
    "PARP1": 100,
    "PGR": 100,
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
LOGGING_EVAL_FREQ_MAP = {
    "pcg": 1,
    "sap": 100,
    "sdd": 100,
}
LOGGING_EVAL_FREQ_MAP_TAXI = {
    "pcg": 1,
    "sap": 200,
    "sdd": 200,
}
