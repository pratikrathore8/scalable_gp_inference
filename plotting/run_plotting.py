"""
Run All Plotting Workflows
--------------------------
This script executes the plotting pipeline for all datasets and x-axis options.
"""

from datetime import datetime
from pathlib import Path
from wandb_utils import (
    get_project_runs,
    filter_runs,
    choose_runs,
    organize_runs_data,
    choose_and_aggregate_runs,
    aggregate_runs_with_stats
    )
from plotting_utils import Plotter
from wandb_constants import (
    DATASETS,
    METRIC_PATHS,
    STRATEGY_MAP,
    SOLVERS,
    CONFIG_KEYS,
    X_AXIS_OPTIONS,
    BAYESIAN_OPT_CONFIG_KEYS,
    BAYESIAN_OPT_METRIC_PATHS,
    BAYESIAN_OPT_X_AXIS_OPTIONS,
    LENGTHSCALES
)
from plotting_constants import BASE_SAVE_DIR, BAYESIAN_OPT_BASE_SAVE_DIR


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ENTITY = "sketchy-opts"


class PlottingConfig:
    """
    Configuration class to handle running plotting functions, taking into account the differences between
    Bayesian optimization and other tasks
    """
    def __init__(self, is_bayesian_opt: bool):
        self.is_bayesian_opt = is_bayesian_opt

        if is_bayesian_opt:
            self.config_keys = BAYESIAN_OPT_CONFIG_KEYS
            self.metric_paths = BAYESIAN_OPT_METRIC_PATHS
            self.x_axis_options = BAYESIAN_OPT_X_AXIS_OPTIONS
            self.base_save_dir = BAYESIAN_OPT_BASE_SAVE_DIR
            self.project_prefix = "bayesopt_lengthscale_"
            self.item_type = "kernel lengthscale"
            self.save_dir_prefix = "kernel_lengthscale_"
        else:
            self.config_keys = CONFIG_KEYS
            self.metric_paths = METRIC_PATHS
            self.x_axis_options = X_AXIS_OPTIONS
            self.base_save_dir = BASE_SAVE_DIR
            self.project_prefix = "gp_inference_"
            self.item_type = "dataset"
            self.save_dir_prefix = ""

    def get_project_name(self, item: str) -> str:
        return f"{self.project_prefix}{item}"

    def get_save_dir(self, item: str) -> Path:
        if self.is_bayesian_opt:
            return Path(self.base_save_dir) / f"{self.save_dir_prefix}{item}"
        else:
            return Path(self.base_save_dir) / item

    def get_filter_config(self, item: str, solvers: list[str]) -> dict:
        if self.is_bayesian_opt:
            return {self.config_keys["SOLVER"]: solvers}
        else:
            return {
                self.config_keys["DATASET"]: item,
                self.config_keys["SOLVER"]: solvers
            }

    def format_item_message(self, item: str, message: str) -> str:
        if self.is_bayesian_opt:
            return f"{message} for {self.item_type} {item}"
        else:
            return f"{message} for {item}"


def run_plotting(is_bayesian_opt: bool,
                datasets_or_lengthscales: list[str],
                solvers: list[str],
                strategy_map: dict[str, str],
                y_metrics: list[str],
                best_strategy_metric: str,
                best_strategy_metric_agg: str,
                x_axis_options: dict[str, str]
                ):
    """Plots for all given datasets/lengthscales, y metrics, and x-axis options."""

    config = PlottingConfig(is_bayesian_opt)

    for item in datasets_or_lengthscales:
        project = config.get_project_name(item)
        runs = get_project_runs(ENTITY, project)

        if not runs:
            print(config.format_item_message(item, "No runs available") + " on W&B")
            continue

        try:
            filtered = filter_runs(
                runs,
                require_all=config.get_filter_config(item, solvers)
            )
        except KeyError as e:
            print(config.format_item_message(item, f"Error filtering: {str(e)}"))
            continue

        if not filtered:
            print(config.format_item_message(item, "No valid runs after filtering"))
            continue

        try:
            selected_runs = choose_runs(
                is_bayesian_opt,
                filtered,
                strategy_map,
                best_strategy_metric,
                best_strategy_metric_agg
            )
        except Exception as e:
            print(config.format_item_message(item, f"Error selecting runs: {str(e)}"))
            continue

        if not selected_runs:
            print(config.format_item_message(item, "No selected runs"))
            continue

        for x_axis in x_axis_options.values():
            runs_data = organize_runs_data(
                is_bayesian_opt,
                selected_runs,
                y_metrics,
                x_axis
            )

            if not runs_data:
                print(config.format_item_message(item, f"No plottable data (x-axis: {x_axis})"))
                continue

            plotter = Plotter(runs_data, aggregated=False, is_bayesian_opt=is_bayesian_opt)

            save_dir = config.get_save_dir(item)
            save_dir.mkdir(parents=True, exist_ok=True)

            title = None if is_bayesian_opt else item

            for y_metric in y_metrics:
                y_key = [key for key, value in config.metric_paths.items() if value == y_metric][0]
                plotter.plot_single_metric(
                    y_metric=y_metric,
                    x_axis=x_axis,
                    log_y=True,
                    title=title,
                    save_path=save_dir / f"{y_key}_{x_axis}_{item}_{TIMESTAMP}"
                )

            plotter.plot_metric_grid(
                y_metrics=y_metrics,
                x_axis=x_axis,
                log_y=True,
                title=title,
                save_path=save_dir / f"MULTI_METRIC_{x_axis}_{item}_{TIMESTAMP}"
            )


def run_all(is_bayesian_opt: bool | None):
    """
    Run plotting for all datasets, all solvers, all metrics, and all x-axis options.

    If is_bayesian_opt is None, it runs plottings for both Bayesian optimization task and datasets.
    """

    def run_single_task(is_opt: bool):
        if is_opt:
            run_plotting(
                is_bayesian_opt=True,
                datasets_or_lengthscales=LENGTHSCALES,
                solvers=SOLVERS,
                strategy_map=STRATEGY_MAP,
                y_metrics=[BAYESIAN_OPT_METRIC_PATHS["FN_MAX"]],
                best_strategy_metric=BAYESIAN_OPT_METRIC_PATHS["FN_MAX"],
                best_strategy_metric_agg="last",
                x_axis_options=BAYESIAN_OPT_X_AXIS_OPTIONS
            )
        else:
            run_plotting(
                is_bayesian_opt=False,
                datasets_or_lengthscales=DATASETS,
                solvers=SOLVERS,
                strategy_map=STRATEGY_MAP,
                y_metrics=[METRIC_PATHS["TEST_RMSE"], METRIC_PATHS["POSTERIOR_MEAN_NLL"]],
                best_strategy_metric=METRIC_PATHS["TEST_RMSE"],
                best_strategy_metric_agg="last",
                x_axis_options=X_AXIS_OPTIONS
            )

    if is_bayesian_opt is None:
        run_single_task(True)
        run_single_task(False)
    else:
        run_single_task(is_bayesian_opt)


def run_errorbars(is_bayesian_opt: bool,
                  datasets_or_lengthscales: list[str],
                  solvers: list[str],
                  y_metrics: list[str],
                  num_seeds: int,
                  sort_metric: str):
    """Plots error bars for the given datasets/lengthscales, y metrics, and solvers for a given number of random seeds."""

    config = PlottingConfig(is_bayesian_opt)

    for item in datasets_or_lengthscales:
        project = config.get_project_name(item)
        runs = get_project_runs(ENTITY, project)

        if not runs:
            print(config.format_item_message(item, "No runs available") + " on W&B")
            continue

        try:
            filtered = filter_runs(
                runs,
                require_all=config.get_filter_config(item, solvers)
            )
        except KeyError as e:
            print(config.format_item_message(item, f"Error filtering: {str(e)}"))
            continue

        if not filtered:
            print(config.format_item_message(item, "No valid runs after filtering"))
            continue

        try:
            agg_data = choose_and_aggregate_runs(is_bayesian_opt, filtered, y_metrics, num_seeds, sort_metric)
        except Exception as e:
            print(config.format_item_message(item, f"Error selecting runs: {str(e)}"))
            continue

        if not agg_data:
            print(config.format_item_message(item, "No selected runs"))
            continue

        bar_plotter = Plotter(agg_data, aggregated=True, is_bayesian_opt=is_bayesian_opt)

        save_dir = config.get_save_dir(item)
        save_dir.mkdir(parents=True, exist_ok=True)

        title = None if is_bayesian_opt else item

        for y_metric in y_metrics:
            y_key = [key for key, value in config.metric_paths.items() if value == y_metric][0]
            bar_plotter.plot_errorbars(
                metric=y_metric,
                title=title,
                save_path=save_dir / f"ERRORBAR_{y_key}_{item}_{TIMESTAMP}"
            )


def run_with_errorbands(
    is_bayesian_opt: bool,
    datasets_or_lengthscales: list[str],
    solvers: list[str],
    y_metrics: list[str],
    x_axis_options: dict[str, str],
    num_runs: int,
    sort_metric: str = None,
    group_by: list[str] = None
):
    """
    Plot metrics with error bands from multiple runs, using the top performing runs.
    """
    if group_by is None:
        group_by = []

    if sort_metric is None and y_metrics:
        sort_metric = y_metrics[0]

    config = PlottingConfig(is_bayesian_opt)
    task_type = "Bayesian optimization task" if is_bayesian_opt else "dataset"

    for item in datasets_or_lengthscales:
        project = config.get_project_name(item)
        runs = get_project_runs(ENTITY, project)

        if not runs:
            print(config.format_item_message(item, f"No {task_type} runs available") + " on W&B")
            continue

        try:
            filtered = filter_runs(
                runs,
                require_all={CONFIG_KEYS["SOLVER"]: solvers}
            )
        except KeyError as e:
            print(f"Error filtering: {str(e)}")
            continue

        if not filtered:
            print(config.format_item_message(item, f"No valid {task_type} runs after filtering"))
            continue

        for x_axis_key, x_axis in x_axis_options.items():
            print(f"Processing {config.format_item_message(item, f'{task_type}')} with x-axis {x_axis_key}...")

            aggregated_data = aggregate_runs_with_stats(
                is_bayesian_opt,
                filtered,
                y_metrics,
                x_axis,
                num_runs=num_runs,
                sort_metric=sort_metric,
                group_by=group_by
            )

            if not aggregated_data:
                print(config.format_item_message(item, f"No valid groups {task_type} (x-axis: {x_axis})"))
                continue

            plotter = Plotter({}, aggregated=False, is_bayesian_opt=is_bayesian_opt)

            save_dir = config.get_save_dir(item)
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            title = None
            prefix = "Bayesian" if is_bayesian_opt else item

            for y_key, y_metric in [(k, v) for k, v in config.metric_paths.items() if v in y_metrics]:
                plotter.plot_with_errorbands(
                    aggregated_data,
                    y_metric=y_metric,
                    x_axis=x_axis,
                    log_y=True,
                    title=title,
                    save_path=save_dir / f"ERRORBAND_{prefix}_{y_key}_{x_axis_key}_{timestamp}"
                )

            plotter.plot_metric_grid_with_errorbands(
                aggregated_data,
                y_metrics=y_metrics,
                x_axis=x_axis,
                log_y=True,
                title=title,
                save_path=save_dir / f"ERRORBAND_MULTI_{prefix}_{x_axis_key}_{timestamp}"
            )

            print(f"Completed {config.format_item_message(item, f'{task_type}')} with x-axis {x_axis_key}")


# Example Usage
if __name__ == "__main__":
    run_plotting(
        is_bayesian_opt=False,
        datasets_or_lengthscales=DATASETS,
        solvers=["sap", "pcg", "sdd"],
        strategy_map={"sap": "best", "pcg": "best", "sdd": "best"},
        y_metrics=[METRIC_PATHS["TEST_RMSE"], METRIC_PATHS["POSTERIOR_MEAN_NLL"]],
        best_strategy_metric=METRIC_PATHS["TEST_RMSE"],
        best_strategy_metric_agg="last",
        x_axis_options=X_AXIS_OPTIONS
    )

    run_plotting(
        is_bayesian_opt=True,
        datasets_or_lengthscales=LENGTHSCALES,
        solvers=["sap", "pcg", "sdd"],
        strategy_map={"sap": "best", "pcg": "best", "sdd": "best"},
        y_metrics=[BAYESIAN_OPT_METRIC_PATHS["FN_MAX"]],
        best_strategy_metric=BAYESIAN_OPT_METRIC_PATHS["FN_MAX"],
        best_strategy_metric_agg="last",
        x_axis_options=BAYESIAN_OPT_X_AXIS_OPTIONS
    )

    print("\nRunning error band plotting...")
    run_with_errorbands(
        is_bayesian_opt=False,
        datasets_or_lengthscales=DATASETS,
        solvers=["sap", "pcg", "sdd"],
        y_metrics=[METRIC_PATHS["TEST_RMSE"], METRIC_PATHS["POSTERIOR_MEAN_NLL"]],
        x_axis_options=X_AXIS_OPTIONS,
        num_runs=5,
        sort_metric=METRIC_PATHS["TEST_RMSE"]
    )

    run_with_errorbands(
        is_bayesian_opt=True,
        datasets_or_lengthscales=LENGTHSCALES,
        solvers=["sap", "pcg", "sdd"],
        y_metrics=[BAYESIAN_OPT_METRIC_PATHS["FN_MAX"]],
        x_axis_options=BAYESIAN_OPT_X_AXIS_OPTIONS,
        num_runs=2,
        sort_metric=BAYESIAN_OPT_METRIC_PATHS["FN_MAX"]
    )