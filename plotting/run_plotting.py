"""
Run All Plotting Workflows
--------------------------
This script executes the plotting pipeline for all datasets and x-axis options,
automating the process across the full experimental matrix.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from wandb_utils import get_project_runs, filter_runs, choose_runs, organize_runs_data
from plotting_utils_final import Plotter
from wandb_constants import (
    DATASETS,
    METRIC_PATHS,
    STRATEGY_MAP,
    SOLVERS,
    CONFIG_KEYS,
    X_AXIS_OPTIONS
)
from plotting_constants import BASE_SAVE_DIR

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ENTITY = "sketchy-opts"

def run_plotting(datasets: list[str],
                 solvers: list[str],
                 strategy_map: dict[str, str],
                y_metrics: list[str],
                best_strategy_metric: str,
                best_strategy_metric_agg: str,
                x_axis_options: dict[str, str]
                ):

    """Plots for all given datasets, y metrics, and x-axis options. Returuns both individual and grid plots."""
    for dataset in datasets:
        project = f"gp_inference_{dataset}"
        runs = get_project_runs(ENTITY, project)

        if not runs:
            print(f"No runs available for {dataset} on W&B")
            continue

        # Filter runs
        try:
            filtered = filter_runs(
                runs,
                require_all={
                    CONFIG_KEYS["DATASET"]: dataset,
                    CONFIG_KEYS["SOLVER"]: solvers
                }
            )
        except KeyError as e:
            print(f"Error filtering {dataset}: {str(e)}")
            continue

        if not filtered:
            print(f"No valid runs after filtering for {dataset}")
            continue

        # Select representative runs
        try:
            selected_runs = choose_runs(
                filtered,
                strategy_map,
                best_strategy_metric,
                best_strategy_metric_agg
            )
        except Exception as e:
            print(f"Error selecting runs for {dataset}: {str(e)}")
            continue

        if not selected_runs:
            print(f"No selected runs for {dataset}")
            continue

        # Process each x-axis option
        for x_axis in x_axis_options.values():
            # Organize data for current x-axis
            runs_data = organize_runs_data(
                selected_runs,
                y_metrics,
                x_axis
            )

            if not runs_data:
                print(f"No plottable data for {dataset} (x-axis: {x_axis})")
                continue

            # Initialize plotter and save paths
            plotter = Plotter(runs_data)

            save_dir = Path(BASE_SAVE_DIR) / dataset
            save_dir.mkdir(parents=True, exist_ok=True)


            for y_metric in y_metrics:
                # Generate single metric plot
                y_key = [key for key, value in METRIC_PATHS.items() if value == y_metric][0]
                plotter.plot_single_metric(
                    y_metric=y_metric,
                    x_axis=x_axis,
                    title=dataset,
                    save_path=save_dir / f"{y_key}_{x_axis}_{dataset}_{TIMESTAMP}"
                )

            # Generate metric grid plot
            plotter.plot_metric_grid(
                y_metrics=y_metrics,
                x_axis=x_axis,
                title=dataset,
                save_path=save_dir / f"MULTI_METRIC_{x_axis}_{dataset}_{TIMESTAMP}"
            )


def run_all():
    """Run plotting for all datasets, all solvers, all metrics, and all x-axis options.
    The strategy map is set to 'best' for all solvers by default in plotting_constants.py.
    The best strategy metric is set to TEST_RMSE by default in plotting_constants.py."""
    for dataset in DATASETS:
        project = f"gp_inference_{dataset}"
        runs = get_project_runs(ENTITY, project)

        if not runs:
            print(f"No runs available for {dataset} on W&B")
            continue

        # Filter runs
        try:
            filtered = filter_runs(
                runs,
                require_all={
                    CONFIG_KEYS["DATASET"]: dataset,
                    CONFIG_KEYS["SOLVER"]: SOLVERS
                }
            )
        except KeyError as e:
            print(f"Error filtering {dataset}: {str(e)}")
            continue

        if not filtered:
            print(f"No valid runs after filtering for {dataset}")
            continue

        # Select representative runs
        try:
            selected_runs = choose_runs(
                filtered,
                strategy_map=STRATEGY_MAP,
                metric=METRIC_PATHS["TEST_RMSE"],
                metric_agg="last"
            )
        except Exception as e:
            print(f"Error selecting runs for {dataset}: {str(e)}")
            continue

        if not selected_runs:
            print(f"No selected runs for {dataset}")
            continue

        # Process each x-axis option
        y_metrics = list(METRIC_PATHS.values())
        for x_axis in X_AXIS_OPTIONS.values():
            # Organize data for current x-axis
            runs_data = organize_runs_data(
                selected_runs,
                y_metrics,
                x_axis
            )

            if not runs_data:
                print(f"No plottable data for {dataset} (x-axis: {x_axis})")
                continue

            plotter = Plotter(runs_data)

            save_dir = Path(BASE_SAVE_DIR) / dataset
            save_dir.mkdir(parents=True, exist_ok=True)

            for y_key, y_metric in METRIC_PATHS.items():
                # Generate single metric plot
                plotter.plot_single_metric(
                    y_metric=y_metric,
                    x_axis=x_axis,
                    title=dataset,
                    save_path=save_dir / f"{y_key}_{x_axis}_{dataset}_{TIMESTAMP}"
                )

            # Generate metric grid plot
            plotter.plot_metric_grid(
                y_metrics=y_metrics,
                x_axis=x_axis,
                title=dataset,
                save_path=save_dir / f"MULTI_METRIC_{x_axis}_{dataset}_{TIMESTAMP}"
            )

# Example Usage
if __name__ == "__main__":
    run_plotting(
        datasets=DATASETS,
        solvers=["sap", "pcg", "sdd"],
        strategy_map={"sap": "best", "pcg": "best", "sdd": "best"},
        y_metrics=[METRIC_PATHS["TEST_RMSE"], METRIC_PATHS["POSTERIOR_MEAN_NLL"]],
        best_strategy_metric=METRIC_PATHS["TEST_RMSE"],
        best_strategy_metric_agg="last",
        x_axis_options=X_AXIS_OPTIONS
    )
