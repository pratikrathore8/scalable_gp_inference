from plotting.constants import (
    BASE_SAVE_DIR,
    SAVE_EXTENSION,
    PROJECT_NAME_BASE_BO,
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
)

METRIC = "fn_max"
LENGTHSCALE = 3.0
X_AXIS_NAME = "num_acquisitions"


def _do_plotting_for_metric(lengthscale: float, metric: str):
    runs = get_runs(PROJECT_NAME_BASE_BO + str(lengthscale), mode="bo")
    max_num_acquisitions = max([run.run.summary["num_acquisitions"] for run in runs])

    # Only keep runs with the maximum number of acquisitions -- this will exclude runs
    # that were unstable and ended up stopping early
    runs = [
        run
        for run in runs
        if run.run.summary["num_acquisitions"] == max_num_acquisitions
    ]

    # Split runs based on seeds
    runs_split = {}
    for run in runs:
        seed_key = f"Seed {run.run.config['seed']}"
        if seed_key not in runs_split:
            runs_split[seed_key] = []
        runs_split[seed_key].append(run)

    statistics_dicts = {}
    colors_dict = {}
    for seed_key, runs in runs_split.items():
        metrics_dict, colors_dict_temp = get_metrics_and_colors(runs, metric)
        colors_dict.update(
            colors_dict_temp
        )  # update the colors_dict with the new colors to cover all optimizers
        statistics_dicts[seed_key] = get_metric_statistics(metrics_dict)

    save_path = get_save_path(
        f"{BASE_SAVE_DIR}/bo/lengthscale_{lengthscale}",
        f"{metric}_{X_AXIS_NAME}.{SAVE_EXTENSION}",
    )

    # Plot the statistics
    plot_metric_statistics(
        statistics_dicts,
        colors_dict,
        X_AXIS_NAME,
        use_min_time=False,
        save_path=save_path,
    )


if __name__ == "__main__":
    # Set the font size for the plots
    set_fontsize(FONTSIZE)

    # Render in LaTeX
    render_in_latex()

    _do_plotting_for_metric(LENGTHSCALE, METRIC)
