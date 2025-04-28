from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Global LaTeX / fonts / file-format
# ──────────────────────────────────────────────────────────────────────────────
USE_LATEX: bool = False          # call `mpl.rcParams.update({"text.usetex": ...})`
FONTSIZE:  int  = 20
EXTENSION: str  = "pdf"         # default file suffix for figures
BASE_SAVE_DIR: str = "./plots"  # created on demand

# ──────────────────────────────────────────────────────────────────────────────
# Experiment axes / hyper-parameter labelling
# ──────────────────────────────────────────────────────────────────────────────
X_AXIS: str = "time"            # mnemonic used by helper functions

# Which hyper-parameters are worth showing in legend labels for each solver
HPARAMS_TO_LABEL: dict[str, list[str]] = {
    # PCG exposes its Nyström preconditioner’s details
    "pcg": ["rank", "rho", "damping_mode"],
    # SAP (stochastic average projections) has no exposed HPs in the config
    "sap": [],
}

# ──────────────────────────────────────────────────────────────────────────────
# Figure sizing helpers (in inches, to match Matplotlib defaults)
# ──────────────────────────────────────────────────────────────────────────────
SZ_COL: float = 8.0   # width for a “single column” figure
SZ_ROW: float = 6.0   # height for a “single row”

# ──────────────────────────────────────────────────────────────────────────────
# Legend placement
# ──────────────────────────────────────────────────────────────────────────────
LEGEND_SPECS: dict = dict(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    frameon=False,
)

# ──────────────────────────────────────────────────────────────────────────────
# Colour and marker maps
# ──────────────────────────────────────────────────────────────────────────────
OPT_COLORS: dict[str, str] = {
    "pcg":  cm.get_cmap("Blues")(0.6),
    "sap":  cm.get_cmap("Greens")(0.6),
}

# Nyström-rank is a continuous quantity → use a continuous colour‐map
RANK_MIN, RANK_MAX = 0, 2_000
RANK_NORM = Normalize(vmin=RANK_MIN, vmax=RANK_MAX)

# Discrete markers for preconditioning / sampling options (extend if needed)
PRECOND_MARKERS: dict[str, dict] = {
    "nystrom":      {"adaptive": "o", "fixed": "x"},
}

SAMPLING_LINESTYLES: dict[str, str] = {
    "uniform": "solid",
    "rls":     "dashed",
}

TOT_MARKERS: int = 10     # how many points in a line should be rendered as markers
MARKERSIZE: int = 8

# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
METRIC_AX_PLOT_FNS: dict[str, str] = {
    "abs_res":                       "semilogy",
    "rel_res":                       "semilogy",
    "train_rmse":                    "semilogy",
    "test_rmse":                     "semilogy",
    "test_posterior_samples_nll":    "semilogy",
    "test_posterior_samples_mean":   "plot",
    "test_posterior_samples_var":    "plot",
}

# ──────────────────────────────────────────────────────────────────────────────
# Metric labels
# ──────────────────────────────────────────────────────────────────────────────
METRIC_LABELS: dict[str, str] = {
    "abs_res":                       r"Absolute residual",
    "rel_res":                       r"Relative residual",
    "train_rmse":                    r"Train RMSE",
    "test_rmse":                     r"Test RMSE",
    "test_posterior_samples_nll":    r"NLL (posterior samples)",
    "test_posterior_samples_mean":   r"Posterior mean",
    "test_posterior_samples_var":    r"Posterior variance",
}

OPT_LABELS: dict[str, str] = {
    "pcg": r"\texttt{PCG}",
    "sap": r"\texttt{SAP}",
}

# X-axis labels for the three canonical choices
X_AXIS_LABELS: dict[str, str] = dict(
    time="Time (s)",
    datapasses="Full data passes",
    iters="Iterations",
)

# Nan sentinel when a metric is missing
NAN_REPLACEMENT: float = np.inf
