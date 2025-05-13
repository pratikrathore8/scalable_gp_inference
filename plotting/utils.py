import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import wandb

from plotting.constants import (
    BOUND_FILL,
    ENTITY_NAME,
    GRID_FILL,
    LEGEND_SPECS,
    METRIC_NAME_MAP,
    METRIC_YLIMS_MAP,
    SZ_COL,
    SZ_ROW,
    TIMING_PLOT_COLOR,
    X_AXIS_NAME_MAP,
)
from plotting.metric_classes import MetricData
from plotting.run_classes import WandbRun, WandbRunBO


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


def get_runs(project_name: str, mode="gp_inference") -> list[WandbRun]:
    """
    Get all runs from a wandb project.

    Args:
        project_name: Name of the wandb project
        mode: Mode for getting runs. Can be "gp_inference" or "bo".

    Returns:
        List of runs
    """
    api = wandb.Api()
    run_class = WandbRunBO if mode == "bo" else WandbRun
    runs = [run_class(run) for run in api.runs(f"{ENTITY_NAME}/{project_name}")]
    return runs


def get_metrics_and_colors(
    runs: list[WandbRun],
    metric: str,
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
        metric_statistics_class = metrics_list[0].__class__
        if not all(isinstance(m, metric_statistics_class) for m in metrics_list):
            warnings.warn(
                f"Skipping {opt_name} due to inconsistent metric data classes."
            )
            continue
        mean_data, min_data, max_data = metric_statistics_class.compute_statistics(
            metrics_list
        )

        if mean_data is None or min_data is None or max_data is None:
            warnings.warn(
                f"Skipping {opt_name} due to inconsistent metric data across runs."
            )
            continue

        all_statistics[opt_name] = (mean_data, min_data, max_data)
    return all_statistics


def _get_grid_size(
    grid_size: tuple[int, int] | None, n_datasets: int
) -> tuple[int, int]:
    if grid_size is None:
        cols = min(3, n_datasets)
        rows = (n_datasets + cols - 1) // cols  # Ceiling division
    else:
        rows, cols = grid_size
    return rows, cols


def _shape_axes(axes, rows: int, cols: int):
    # Convert to 2D array if needed
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    return axes


def _get_axis(axes, i: int, cols: int):
    row_idx = i // cols
    col_idx = i % cols
    ax = axes[row_idx, col_idx]
    return ax


def _hide_unused_subplots(axes, n_datasets: int, rows: int, cols: int):
    # Hide unused subplots
    for i in range(n_datasets, rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        axes[row_idx, col_idx].axis("off")


def _get_n_sci(n):
    n_sci = re.sub(r"e\+?0*(\d+)", r" \\cdot 10^{\1}", f"{n:.2e}")
    n_sci = re.sub(r"e-0*(\d+)", r" \\cdot 10^{-\1}", n_sci)
    return n_sci


def _get_title(dataset: str, dataset_size: int | None) -> str:
    """Generate a title for the subplot."""
    if dataset_size is not None:
        return f"{dataset} ($n = {_get_n_sci(dataset_size)}$)"
    else:
        return dataset


def _set_ylim(ax, metric_name: str) -> tuple[float, float] | None:
    """Set y-axis limits for a given metric name."""
    ylims = METRIC_YLIMS_MAP.get(metric_name, None)
    if ylims is not None:
        ax.set_ylim(ylims)


def _add_grid(ax):
    """Add grid to the axis."""
    ax.grid(axis="y", linestyle="--", alpha=GRID_FILL)


def _plot_metric_statistics_helper(
    ax,
    statistics_dict,
    colors_dict,
    x_axis_name,
    dataset,
    dataset_size,
    use_min_time,
):
    """Helper function to plot metrics on a given axis."""
    # Initialize min and max values for x axis
    min_final_time = np.inf
    xlims = (np.inf, -np.inf)
    metric_name = None

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

    # Set the second xlim to the minimum final time, if desired
    if x_axis_name == "time" and use_min_time:
        xlims = (xlims[0], min_final_time)

    ax.set_xlim(xlims)
    _set_ylim(ax, metric_name)
    ax.set_xlabel(X_AXIS_NAME_MAP[x_axis_name])
    ax.set_ylabel(METRIC_NAME_MAP[metric_name])
    ax.set_title(_get_title(dataset, dataset_size))

    # Add grid for better readability
    _add_grid(ax)


def plot_metric_statistics(
    statistics_dicts: dict[str, dict[str, tuple[MetricData, MetricData, MetricData]]],
    colors_dict: dict[str, str],
    x_axis_name: str,
    size_dict: dict[str, int] = {},
    use_min_time: bool = True,
    grid_size: tuple[int, int] = None,
    save_path: str = None,
):
    """
    Plot metric statistics for multiple datasets as subplots.

    Args:
        statistics_dicts: Dictionary of dictionaries containing statistics
          for each dataset
        colors_dict: Dictionary mapping optimizer names to colors
        x_axis_name: Name of the x-axis to use
        size_dict: Dictionary mapping dataset names to sizes for the
          subplots.
        use_min_time: If True, use the minimum final time for the x-axis limit
        grid_size: Tuple (rows, cols) for subplot grid. If None,
          will be automatically determined
        save_path: Path to save the figure
    """
    # Determine grid size if not provided
    n_datasets = len(statistics_dicts)
    rows, cols = _get_grid_size(grid_size, n_datasets)

    fig, axes = plt.subplots(rows, cols, figsize=(SZ_COL * cols, SZ_ROW * rows))

    # Convert to 2D array if needed
    axes = _shape_axes(axes, rows, cols)

    # Keep track of unique optimizer names across all datasets
    all_opt_names = set()
    for stats_dict in statistics_dicts.values():
        all_opt_names.update(stats_dict.keys())

    # Create a mapping of optimizer names to line objects for the legend
    legend_handles_dict = {}

    # Plot each dataset
    for i, (dataset, statistics_dict) in enumerate(list(statistics_dicts.items())):
        ax = _get_axis(axes, i, cols)

        dataset_size = size_dict.get(dataset, None)

        _plot_metric_statistics_helper(
            ax,
            statistics_dict,
            colors_dict,
            x_axis_name,
            dataset,
            dataset_size,
            use_min_time,
        )

        # Collect line objects for legend from this plot
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in legend_handles_dict:
                legend_handles_dict[label] = handle

    # Hide unused subplots
    _hide_unused_subplots(axes, n_datasets, rows, cols)

    # Create legend handles and labels in a consistent order
    sorted_opt_names = sorted(legend_handles_dict.keys())
    legend_handles = [legend_handles_dict[name] for name in sorted_opt_names]

    # Add a single legend for the entire figure
    fig.legend(legend_handles, sorted_opt_names, **LEGEND_SPECS)

    plt.tight_layout()
    _savefig(fig, save_path)


def plot_parallel_scaling(
    runs: list[WandbRun],
    save_path: str = None,
):
    """
    Plot the parallel scaling for runs belonging to a given dataset and optimizer.

    Args:
        runs: List of WandbRun objects
        save_path: Path to save the figure
    """
    # NOTE(pratik): This function assumes that the runs are all for the same dataset
    # and that they are all for the same optimizer.
    # If this is not the case, the plot will not make sense.
    fig, ax = plt.subplots(figsize=(SZ_COL, SZ_ROW))

    dataset = runs[0].run.config["dataset"]
    dataset_size = runs[0].run.config["ntr"]

    # Find the run with exactly one device
    reference_time = None
    for run_obj in runs:
        if len(run_obj.run.config["all_devices"]) == 1:
            reference_time = run_obj.run.summary["cum_time"]
    if reference_time is None:
        raise ValueError("No run with exactly one device found.")

    # Calculate speedup
    speedup_dict = {}
    for run_obj in runs:
        num_devices = len(run_obj.run.config["all_devices"])
        speedup = reference_time / run_obj.run.summary["cum_time"]
        speedup_dict[num_devices] = speedup

    # Sort the speedup dictionary by number of devices
    sorted_speedup = dict(sorted(speedup_dict.items()))
    num_devices = list(sorted_speedup.keys())
    speedup = list(sorted_speedup.values())

    # Create a line plot that compares to y == x
    ax.plot(
        num_devices,
        num_devices,
        label="Ideal Speedup",
        linestyle="--",
        color=TIMING_PLOT_COLOR,
    )
    ax.plot(
        num_devices,
        speedup,
        label="Measured Speedup",
        marker="o",
        color=TIMING_PLOT_COLOR,
    )

    # Set x-ticks to only show integer values
    # Get min and max device numbers to set the tick range
    min_devices = min(num_devices)
    max_devices = max(num_devices)

    # Create integer ticks from min to max
    ax.set_xticks(range(min_devices, max_devices + 1))

    # Ensure the x-axis limits cover the data range
    ax.set_xlim(min_devices - 0.5, max_devices + 0.5)

    # Add labels and title
    ax.set_xlabel(X_AXIS_NAME_MAP["devices"])
    ax.set_ylabel("Speedup")
    ax.set_title(_get_title(dataset, dataset_size))

    # Add grid for better readability
    _add_grid(ax)

    fig.legend(**LEGEND_SPECS)

    plt.tight_layout()
    _savefig(fig, save_path)


def _bar_metric_statistics_helper(
    ax,
    statistics_dict,
    colors_dict,
    dataset,
    dataset_size,
):
    """Helper function to create bar plots on a given axis."""
    # Lists to store data for the bar chart
    opt_names = []
    final_values = []
    lower_bounds = []
    upper_bounds = []
    metric_name = None

    for i, (opt_name, statistics) in enumerate(list(statistics_dict.items())):
        mean_data, lower_bound_data, upper_bound_data = statistics

        # Extract name for y-axis
        if i == 0:
            metric_name = mean_data.metric_name

        # Store the final values (last element of each array)
        if mean_data.finished:
            opt_names.append(opt_name)
            final_values.append(mean_data.metric_data[-1])
            lower_bounds.append(lower_bound_data.metric_data[-1])
            upper_bounds.append(upper_bound_data.metric_data[-1])

    # Calculate error bar heights
    lower_errors = np.array(final_values) - np.array(lower_bounds)
    upper_errors = np.array(upper_bounds) - np.array(final_values)
    errors = np.vstack([lower_errors, upper_errors])

    # Create bar positions
    x_pos = np.arange(len(opt_names))

    # Create the bar chart
    ax.bar(
        x_pos,
        final_values,
        width=0.7,
        align="center",
        yerr=errors,
        capsize=8,
        color=[colors_dict[name] for name in opt_names],
        edgecolor="black",
        linewidth=1.5,
        error_kw={"elinewidth": 2, "capthick": 2},
    )

    # Add labels and title
    ax.set_ylabel(METRIC_NAME_MAP[metric_name])
    _set_ylim(ax, metric_name)
    ax.set_title(_get_title(dataset, dataset_size))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(opt_names, rotation=45, ha="right")

    # Add grid for better readability
    _add_grid(ax)


def bar_metric_statistics(
    statistics_dicts: dict[str, dict[str, tuple[MetricData, MetricData, MetricData]]],
    colors_dict: dict[str, str],
    size_dict: dict[str, int] = {},
    grid_size: tuple[int, int] = None,
    save_path: str = None,
):
    """
    Create bar charts for multiple datasets as subplots.

    Args:
        statistics_dicts: Dictionary of dictionaries containing statistics
          for each dataset
        colors_dict: Dictionary mapping optimizer names to colors
        size_dict: Dictionary mapping dataset names to sizes for the
          subplots.
        grid_size: Tuple (rows, cols) for subplot grid. If None,
          will be automatically determined
        save_path: Path to save the figure
    """
    # Determine grid size if not provided
    n_datasets = len(statistics_dicts)
    rows, cols = _get_grid_size(grid_size, n_datasets)

    fig, axes = plt.subplots(rows, cols, figsize=(SZ_COL * cols, SZ_ROW * rows))

    # Convert to 2D array if needed
    axes = _shape_axes(axes, rows, cols)

    # Plot each dataset
    for i, (dataset, statistics_dict) in enumerate(list(statistics_dicts.items())):
        ax = _get_axis(axes, i, cols)

        dataset_size = size_dict.get(dataset, None)

        _bar_metric_statistics_helper(
            ax, statistics_dict, colors_dict, dataset, dataset_size
        )

    # Hide unused subplots
    _hide_unused_subplots(axes, n_datasets, rows, cols)

    plt.tight_layout()
    _savefig(fig, save_path)
