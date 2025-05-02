"""
Simple Plotting Pipeline Demonstration
-----------------------------------
This script demonstrates the full workflow for fetching, filtering, and visualizing
W&B experiment results using our plotting utility functions.
"""
from datetime import datetime
from pathlib import Path
from wandb_utils import get_project_runs, filter_runs, choose_runs, organize_runs_data
from plotting_utils import Plotter
from wandb_constants import (
    METRIC_PATHS,
    CONFIG_KEYS,
    X_AXIS_OPTIONS,
)
from plotting_constants import BASE_SAVE_DIR

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------------- Data Preparation ---------------------------- #
# Initialize connection to W&B project
ENTITY = "sketchy-opts"
DATASET = "acsincome"
PROJECT = f"gp_inference_{DATASET}"
runs = get_project_runs(ENTITY, PROJECT)

# Filter runs to specific dataset and solvers
filtered = filter_runs(
    runs,
    require_all={
        CONFIG_KEYS["DATASET"]: "acsincome",
        CONFIG_KEYS["SOLVER"]: ["sap", "pcg"] 
    }
)

# Select representative runs per solver
selected_runs = choose_runs(
    filtered,
    strategy_map={
        "sap": "best",
        "pcg": "latest"
    },
    metric=METRIC_PATHS["POSTERIOR_NLL"],
    metric_agg="last"
)

# Organize data for plotting
runs_data = organize_runs_data(
    selected_runs, 
    [METRIC_PATHS["TEST_RMSE"], METRIC_PATHS["POSTERIOR_NLL"]]
)


# ---------------------------- Visualization ---------------------------- #
plotter = Plotter(runs_data)

# Single metric plot
plotter.plot_single_metric(
    y_metric=METRIC_PATHS["POSTERIOR_NLL"],
    x_axis=X_AXIS_OPTIONS["TIME"],
    save_path=Path(BASE_SAVE_DIR) / DATASET / f"POSTERIOR_NLL_{TIMESTAMP}", 
)

# Metric grid plot
plotter.plot_metric_grid(
    y_metrics=[METRIC_PATHS["TEST_RMSE"], METRIC_PATHS["POSTERIOR_NLL"]],
    save_path=Path(BASE_SAVE_DIR) / DATASET / f"TEST_RMSE_POSTERIOR_NLL_{TIMESTAMP}",
)