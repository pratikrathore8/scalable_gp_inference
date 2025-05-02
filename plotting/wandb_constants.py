from typing import Dict, List
# ──────────────────────────────────────────────────────────────────────────────
ENTITY = "sketchy-opts"

DATASETS = [ "ESR2", 
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
            "houseelec" ]  

PROJECT = "gp_inference_acsincome" # Example project name
# ──────────────────────────────────────────────────────────────────────────────
METRIC_PATHS = {
    # Time metrics
    "STEP": "_step",
    "CUM_TIME": "cum_time",
    "ITER_TIME": "iter_time",
    
    # Core evaluation metrics
    "TEST_RMSE": "metrics.callback.test_rmse",
    "TEST_R2": "metrics.callback.test_r2",
    "POSTERIOR_NLL": "metrics.callback.test_posterior_samples_nll",
    "POSTERIOR_MEAN_NLL": "metrics.callback.test_posterior_samples_mean_nll",
    
    # Histogram metrics (require special handling)
    "POSTERIOR_MEAN": "metrics.callback.test_posterior_samples_mean",
    "POSTERIOR_VAR": "metrics.callback.test_posterior_samples_var"
}
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────

SOLVERS = ["sap", "nsap", "pcg", "sdd", "mimosa", "falkon"]
KERNEL_TYPES = ["rbf", "matern"]  # Example values
PRECOND_TYPES = ["nystrom", "partial_cholesky"]

# ──────────────────────────────────────────────────────────────────────────────
HPARAM_LABELS = {
    "sdd": ["precond", "r", "sampling_method"],
    "sap": ["b"],
    "nsap": ["b"],
    "pcg": ["precond", "r"],
    "mimosa": ["precond", "r", "m"],
    "falkon": ["m"]
}

# ──────────────────────────────────────────────────────────────────────────────
X_AXIS_OPTIONS = {
    "STEP": "step",
    "TIME": "time",
    "DATAPASSES": "datapasses"
}

# ──────────────────────────────────────────────────────────────────────────────
# Example Usage Section
# ──────────────────────────────────────────────────────────────────────────────
"""
Example Usage in Analysis Script:

from wandb_utils import get_project_runs, filter_runs, organize_runs_data
from wandb_constants import *

# Fetch recent SAP runs on ACS Income dataset
runs = get_project_runs(ENTITY, PROJECT, max_runs=50)
filtered = filter_runs(
    runs,
    require_all={
        CONFIG_KEYS["DATASET"]: DATASETS[0],
        CONFIG_KEYS["SOLVER"]: "sap"
    }
)

# Organize RMSE data with time axis
data = organize_runs_data(
    filtered,
    y_metrics=[METRIC_PATHS["TEST_RMSE"]],
    x_axis=X_AXIS_OPTIONS["TIME"]
)

# Access results for first run
run_id, df = next(iter(data.items()))
print(f"Final RMSE: {df[METRIC_PATHS['TEST_RMSE']].iloc[-1]:.4f}")
print(f"Time range: {df['x_value'].min():.1f}-{df['x_value'].max():.1f}s")
"""