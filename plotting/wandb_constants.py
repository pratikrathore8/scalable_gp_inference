from typing import Dict, List

ENTITY = "sketchy-opts"

DATASETS = ["ESR2",
            "F2",
            "KIT",
            "PARP1",
            "PGR",
            "acsincome",
            "yolanda",
            "malonaldehyde",
            "benzene",
            "taxi",
            "3droad",
            "song",
            "houseelec"]


METRIC_PATHS = {
    "STEP": "_step",
    "CUM_TIME": "cum_time",
    "ITER_TIME": "iter_time",

    "TEST_RMSE": "metrics.callback.test_rmse",
    "TEST_R2": "metrics.callback.test_r2",
    "POSTERIOR_NLL": "metrics.callback.test_posterior_samples_nll",
    "POSTERIOR_MEAN_NLL": "metrics.callback.test_posterior_samples_mean_nll",

    "POSTERIOR_MEAN": "metrics.callback.test_posterior_samples_mean",
    "POSTERIOR_VAR": "metrics.callback.test_posterior_samples_var"
}

CONFIG_KEYS = {
    "DATASET": "dataset",
    "SOLVER": "solver_name",
    "KERNEL": "kernel_type",
    "NUM_FEATURES": "rf_config.num_features",
    "NUM_TRAIN": "ntr",
    "NUM_TEST": "ntst",
    "NUM_SAMPLES": "num_posterior_samples",
    "BLOCK_SIZE": "solver_config.blk_sz",
    "MAX_ITERS": "solver_config.max_iters"
}

SOLVERS = ["sap", "nsap", "pcg", "sdd", "mimosa", "falkon"]
KERNEL_TYPES = ["rbf", "matern"]
PRECOND_TYPES = ["nystrom", "partial_cholesky"]

HPARAM_LABELS = {
    "sdd": ["precond", "r", "sampling_method"],
    "sap": ["b"],
    "nsap": ["b"],
    "pcg": ["precond", "r"],
    "mimosa": ["precond", "r", "m"],
    "falkon": ["m"]
}

X_AXIS_OPTIONS = {
    "STEP": "step",
    "TIME": "time",
    "DATAPASSES": "datapasses"
}