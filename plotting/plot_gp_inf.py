from plotting.constants import (
    BASE_SAVE_DIR,
    SAVE_EXTENSION,
    PROJECT_NAME_BASE,
    FONTSIZE,
)
from plotting.utils import (
    render_in_latex,
    set_fontsize,
    get_save_path,
    get_runs,
    get_metrics_and_colors,
    get_metric_statistics,
    plot_metric_statistics,
    bar_metric_statistics,
)

DATASETS = ["yolanda", "song", "benzene", "malonaldehyde", "acsincome", "houseelec"]
METRICS = ["test_rmse", "test_posterior_samples_mean_nll"]
X_AXIS_NAME = "time"


def _do_plotting_for_metric(
    datasets: list[str],
    metrics: list[str],
    grid_size: tuple[int, int],
    subfolder_name: str,
):
    """
    Plot the metric for the given datasets.
    """
    runs_dict = {}
    statistics_dicts = {}
    size_dict = {}
    colors_dict = {}

    # Extract the runs for each dataset
    # and the number of training points
    for dataset in datasets:
        runs = get_runs(PROJECT_NAME_BASE + dataset, mode="gp_inference")
        runs_dict[dataset] = runs
        size_dict[dataset] = runs[0].run.config["ntr"]

    # Get statistics for each metric and dataset
    for metric in metrics:
        for dataset in datasets:
            metrics_dict, colors_dict_temp = get_metrics_and_colors(
                runs_dict[dataset], metric
            )
            colors_dict.update(
                colors_dict_temp
            )  # update the colors_dict with the new colors to cover all optimizers
            statistics_dicts[dataset + "_" + metric] = get_metric_statistics(
                metrics_dict
            )

    metrics_combined = "_".join(metrics)

    save_path_plot = get_save_path(
        f"{BASE_SAVE_DIR}/gp_inf/{subfolder_name}",
        f"{metrics_combined}_{X_AXIS_NAME}.{SAVE_EXTENSION}",
    )
    save_path_bar = get_save_path(
        f"{BASE_SAVE_DIR}/gp_inf/{subfolder_name}",
        f"{metrics_combined}_bar.{SAVE_EXTENSION}",
    )

    # Plot the statistics
    plot_metric_statistics(
        statistics_dicts,
        colors_dict,
        X_AXIS_NAME,
        size_dict=size_dict,
        use_min_time=True,
        grid_size=grid_size,
        save_path=save_path_plot,
    )
    bar_metric_statistics(statistics_dicts, colors_dict, save_path=save_path_bar)


if __name__ == "__main__":
    # Set the font size for the plots
    set_fontsize(FONTSIZE)

    # Render in LaTeX
    render_in_latex()

    # Plot the metrics for all datasets (besides taxi)
    # for metric in METRICS:
    _do_plotting_for_metric(
        ["benzene", "malonaldehyde", "houseelec"], METRICS, (2, 3), "all_main"
    )
    _do_plotting_for_metric(
        ["yolanda", "song", "acsincome"], METRICS, (2, 3), "all_appendix"
    )

    # Make plot for intro
    _do_plotting_for_metric(["houseelec"], METRICS, (1, 2), "intro")

    # Also plot for taxi
    _do_plotting_for_metric(["taxi"], ["test_rmse"], (1, 1), "taxi")
