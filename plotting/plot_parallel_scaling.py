from constants import BASE_SAVE_DIR, SAVE_EXTENSION, PROJECT_NAME_BASE, FONTSIZE
from utils import (
    render_in_latex,
    set_fontsize,
    get_save_path,
    get_runs,
    plot_parallel_scaling,
)

DATASET = "taxi"
# We use this for compatibility, but it is not actually used for anything
METRIC = "test_rmse"

if __name__ == "__main__":
    # Set the font size for the plots
    set_fontsize(FONTSIZE)

    # Render in LaTeX
    render_in_latex()

    # Get the runs for the dataset
    runs = get_runs(PROJECT_NAME_BASE + DATASET + "_timing", mode="gp_inference")

    # Plot the timing
    save_path = get_save_path(
        f"{BASE_SAVE_DIR}/parallel_scaling", f"{DATASET}.{SAVE_EXTENSION}"
    )
    plot_parallel_scaling(runs, save_path)
