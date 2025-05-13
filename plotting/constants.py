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
X_AXIS_NAME_MAP = {
    "datapasses": "Datapasses",
    "iterations": "Iterations",
    "time": "Time (s)",
    "num_acquisitions": "Number of Acquisitions",
}
BOUND_FILL = 0.2
FONTSIZE = 20
BASE_SAVE_DIR = "./plots"
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
