import warnings

import matplotlib.pyplot as plt
import numpy as np
import wandb

from plotting.constants import ENTITY_NAME
from plotting.metric_classes import MetricData, WandbRun


def render_in_latex():
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


def set_fontsize(fontsize):
    plt.rcParams.update({"font.size": fontsize})


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
