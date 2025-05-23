ENTITY_NAME = "sketchy-opts"
PROJECT_NAME_BASE = "gp_inference_"
PROJECT_NAME_BASE_BO = "bayesopt_lengthscale_"
METRIC_NAME_BASE = "metrics.callback."
METRIC_NAME_MAP = {
    "test_rmse": "Test RMSE",
    "test_posterior_samples_mean_nll": "Test Mean NLL",
    "train_rmse": "Train RMSE",
    "fn_max": "Max Value",
}
METRIC_YLIMS_MAP = {
    "test_rmse": (0, 2),
    "train_rmse": (0, 2),
}
X_AXIS_NAME_MAP = {
    "datapasses": "Datapasses",
    "iterations": "Iterations",
    "time": "Time (s)",
    "num_acquisitions": "Number of Acquisitions",
    "num_gpus": "Number of GPUs",
}
TIMING_PLOT_COLOR = "tab:blue"
BOUND_FILL = 0.2
GRID_FILL = 0.7
FONTSIZE = 20
BASE_SAVE_DIR = "./figs"
SAVE_EXTENSION = "pdf"
LEGEND_SPECS = {
    "loc": "upper center",
    "bbox_to_anchor": (0.5, 0.0),
    "ncol": 3,
    "frameon": False,
}

# figure size
SZ_COL = 8
SZ_ROW = 6
