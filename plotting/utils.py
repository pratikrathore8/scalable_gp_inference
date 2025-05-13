import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import wandb

from plotting.constants import (
    BOUND_FILL,
    ENTITY_NAME,
    LEGEND_SPECS,
    SZ_COL,
    SZ_ROW,
    X_AXIS_NAME_MAP,
)
from plotting.metric_classes import MetricData, WandbRun


def render_in_latex():
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


def set_fontsize(fontsize):
    plt.rcParams.update({"font.size": fontsize})


def get_save_path(save_dir, save_name):
    if save_dir is not None and save_name is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return os.path.join(save_dir, save_name)
    elif save_dir is not None and save_name is None:
        warnings.warn(
            "Must provide save_name if save_dir is provided. Plot will not be saved."
        )
        return None
    elif save_dir is None and save_name is not None:
        warnings.warn(
            "Must provide save_dir if save_name is provided. Plot will not be saved."
        )
        return None
    else:
        return None


def _savefig(fig, save_path):
    """
    Save the figure to the specified path.

    Args:
        fig: The figure to save
        save_path: The path to save the figure to
    """
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        fig.show()


def get_runs(project_name: str) -> list[WandbRun]:
    """
    Get all runs from a wandb project.

    Args:
        project_name: Name of the wandb project

    Returns:
        List of runs
    """
    api = wandb.Api()
    runs = [WandbRun(run) for run in api.runs(f"{ENTITY_NAME}/{project_name}")]
    return runs


def get_metrics_and_colors(
    runs: list[WandbRun], metric: str
) -> tuple[dict[str, list[MetricData]], dict[str, str]]:
    """
    Get metrics and colors for each run.

    Args:
        runs: List of WandbRun objects
        metric: Metric name to extract
    Returns:
        Tuple of (metrics_dict, colors_dict)
    """
    metrics_dict = {}
    colors_dict = {}

    for run_obj in runs:
        opt_name = run_obj.opt_name
        if opt_name not in metrics_dict:
            metrics_dict[opt_name] = []
            colors_dict[opt_name] = run_obj.color

        # Get metrics for the run
        metric_data = run_obj.get_metric_data(metric)
        metrics_dict[opt_name].append(metric_data)
    return metrics_dict, colors_dict


def compute_metric_statistics(
    metrics_list: list[MetricData],
) -> tuple[MetricData, MetricData, MetricData]:
    """
    Compute mean, minimum, and maximum for a list of MetricData objects.

    Args:
        metrics_list: List of MetricData objects

    Returns:
        Tuple of (mean_data, min_data, max_data)
    """
    # Check if all metrics have the same metric name
    reference = metrics_list[0]
    all_same_name = all(m.metric_name == reference.metric_name for m in metrics_list)
    if not all_same_name:
        raise ValueError("All MetricData objects must have the same metric name")

    # Check if all metrics have the same steps
    reference = metrics_list[0]
    all_same_steps = all(np.array_equal(m.steps, reference.steps) for m in metrics_list)

    if not all_same_steps:
        # raise ValueError("All MetricData objects must have the same steps")
        warnings.warn(
            "Not all MetricData objects have the same steps. "
            "This may lead to incorrect results. "
            "This is likely because some runs were not finished. "
            "We will return None for the mean, min, and max data."
        )
        # Return None for mean, min, and max data
        return None, None, None

    # Stack metric data along a new axis
    stacked_metrics = np.stack([m.metric_data for m in metrics_list], axis=0)

    # Stack cum_times data
    stacked_cum_times = np.stack([m.cum_times for m in metrics_list], axis=0)

    # Compute means
    mean_metrics = np.mean(stacked_metrics, axis=0)
    mean_cum_times = np.mean(stacked_cum_times, axis=0)

    # Find actual min and max values
    min_metrics = np.min(stacked_metrics, axis=0)
    max_metrics = np.max(stacked_metrics, axis=0)

    # Check if all runs are finished
    all_finished = all(m.finished for m in metrics_list)

    # Create MetricData objects for mean, min, and max
    mean_data = MetricData(
        metric_data=mean_metrics,
        steps=reference.steps,
        datapasses=reference.datapasses,
        cum_times=mean_cum_times,
        finished=all_finished,
        metric_name=reference.metric_name,
    )

    min_data = MetricData(
        metric_data=min_metrics,
        steps=reference.steps,
        datapasses=reference.datapasses,
        cum_times=mean_cum_times,  # Using the same mean cum_times for all
        finished=all_finished,
        metric_name=reference.metric_name,
    )

    max_data = MetricData(
        metric_data=max_metrics,
        steps=reference.steps,
        datapasses=reference.datapasses,
        cum_times=mean_cum_times,  # Using the same mean cum_times for all
        finished=all_finished,
        metric_name=reference.metric_name,
    )

    return mean_data, min_data, max_data


def get_metric_statistics(
    metrics_dict: dict[str, list[MetricData]]
) -> dict[str, tuple[MetricData, MetricData, MetricData]]:
    """
    Compute mean, min, and max for each optimizer in the metrics_dict.

    Args:
        metrics_dict: Dictionary of metrics for each optimizer

    Returns:
        Dictionary of mean, min, and max data for each optimizer
    """
    all_statistics = {}
    for opt_name, metrics_list in metrics_dict.items():
        mean_data, min_data, max_data = compute_metric_statistics(metrics_list)

        if mean_data is None or min_data is None or max_data is None:
            warnings.warn(
                f"Skipping {opt_name} due to inconsistent metric data across runs."
            )
            continue

        all_statistics[opt_name] = (mean_data, min_data, max_data)
    return all_statistics


def plot_metric_statistics(
    statistics_dict: dict[str, tuple[MetricData, MetricData, MetricData]],
    colors_dict: dict[str, str],
    x_axis_name: str,
    dataset: str,
    save_path: str = None,
):
    fig, ax = plt.subplots(figsize=(SZ_COL, SZ_ROW))

    # Initialize min and max values for x axis
    min_final_time = np.inf
    xlims = (np.inf, -np.inf)

    for i, (opt_name, statistics) in enumerate(list(statistics_dict.items())):
        mean_data, lower_bound_data, upper_bound_data = statistics

        # Extract name for y-axis
        if i == 0:
            metric_name = mean_data.metric_name

        # Plot the mean and bounds if the runs were all finished
        if mean_data.finished:
            x_axis = mean_data.get_plotting_x_axis(x_axis_name)
            ax.plot(
                x_axis,
                mean_data.metric_data,
                label=opt_name,
                color=colors_dict[opt_name],
            )
            ax.fill_between(
                x_axis,
                lower_bound_data.metric_data,
                upper_bound_data.metric_data,
                color=colors_dict[opt_name],
                alpha=BOUND_FILL,
            )

            # Update xlims
            xlims = (min(xlims[0], x_axis[0]), max(xlims[1], x_axis[-1]))

            # Track the final time for the x-axis limit
            min_final_time = min(min_final_time, mean_data.get_final_time())

    # For time-based x-axis, set the second xlim to the minimum final time
    if x_axis_name == "time":
        xlims = (xlims[0], min_final_time)

    ax.set_xlim(xlims)
    ax.set_xlabel(X_AXIS_NAME_MAP[x_axis_name])
    ax.set_ylabel(metric_name)
    ax.set_title(dataset)
    fig.legend(**LEGEND_SPECS)
    plt.tight_layout()

    _savefig(fig, save_path)
