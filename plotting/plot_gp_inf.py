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


def _do_plotting_for_metric(datasets: list[str], metric: str, subfolder_name: str):
    """
    Plot the metric for the given datasets.
    """
    statistics_dicts = {}
    size_dict = {}

    for dataset in datasets:
        runs = get_runs(PROJECT_NAME_BASE + dataset, mode="gp_inference")
        metrics_dict, colors_dict = get_metrics_and_colors(runs, metric)
        statistics_dicts[dataset] = get_metric_statistics(metrics_dict)
        size_dict[dataset] = runs[0].run.config["ntr"]

    save_path_plot = get_save_path(
        f"{BASE_SAVE_DIR}/gp_inf/{subfolder_name}",
        f"{metric}_{X_AXIS_NAME}.{SAVE_EXTENSION}",
    )
    save_path_bar = get_save_path(
        f"{BASE_SAVE_DIR}/gp_inf/{subfolder_name}",
        f"{metric}_bar.{SAVE_EXTENSION}",
    )

    # Plot the statistics
    plot_metric_statistics(
        statistics_dicts,
        colors_dict,
        X_AXIS_NAME,
        use_min_time=True,
        size_dict=size_dict,
        save_path=save_path_plot,
    )
    bar_metric_statistics(statistics_dicts, colors_dict, save_path=save_path_bar)


if __name__ == "__main__":
    # Set the font size for the plots
    set_fontsize(FONTSIZE)

    # Render in LaTeX
    render_in_latex()

    # Plot the metrics for all datasets (besides taxi)
    for metric in METRICS:
        _do_plotting_for_metric(
            ["benzene", "malonaldehyde", "houseelec"], metric, "all_main"
        )
        _do_plotting_for_metric(
            ["yolanda", "song", "acsincome"], metric, "all_appendix"
        )

    # Also plot for taxi
    _do_plotting_for_metric(["taxi"], "test_rmse", "taxi")
