import matplotlib.cm as cm
import numpy as np
from compressed_root_norm import CompressedRootNorm

USE_LATEX: bool = True
FONTSIZE:  int = 14
EXTENSION: str = "pdf"
BASE_SAVE_DIR: str = "./plots"


HPARAMS_TO_LABEL = {
    "sdd": ["precond", "r", "sampling_method"],
    "sap": ["b"],
    "nsap": ["b"],
    "eigenpro2": [],
    "eigenpro3": ["m"],
    "pcg": ["precond", "r"],
    "falkon": ["m"],
    "mimosa": ["precond", "r", "m"],
}

SZ_COL: float = 8.0
SZ_ROW: float = 6.0

LEGEND_SPECS = {
    "loc": "lower center",
    "bbox_to_anchor": (0.5, -0.35),
    "ncol": 3,
    "frameon": False,
    "fontsize": FONTSIZE * 0.7
}

OPT_COLORS = {
    "sdd": "black",
    "sap": "#FF4500",
    "sap_nystrom": "#FF4500",
    "sap_identity": "#8B0000",
    "nsap": "#4B0082",
    "pcg": "#4169E1",
    "eigenpro2": "tab:pink",
    "eigenpro3": "tab:brown",
    "falkon": "#2E8B57",
    "mimosa": "#778899",
}

RANK_MIN = 0
RANK_MAX = 500 + 1
NORM = CompressedRootNorm(vmin=RANK_MIN, vmax=RANK_MAX, root=3)
DUMMY_PLOTTING_RANK = 100

PRECOND_MARKERS = {
    "sap": {
        "nystrom": "o",
        "identity": "x",
    },
    "nystrom": {"damped": "o", "regularization": "x"},
    "partial_cholesky": {"greedy": "D", "rpc": "v"},
    "falkon": {
        10000: "d",
        20000: "*",
        50000: "s",
        100000: "p",
        200000: "h",
        500000: "8",
        1000000: "+",
    },
}

SAMPLING_LINESTYLES = {
    "uniform": "solid",
    "rls": "dashed",
}

TOT_MARKERS: int = 10
MARKERSIZE: int = 8


METRIC_AX_PLOT_FNS: dict[str, str] = {
    "abs_res":                       "semilogy",
    "rel_res":                       "semilogy",
    "train_rmse":                    "semilogy",
    "test_rmse":                     "semilogy",
    "test_posterior_samples_nll":    "semilogy",
    "test_posterior_samples_mean":   "plot",
    "test_posterior_samples_var":    "plot",
}


METRIC_LABELS: dict[str, str] = {
    "abs_res":                       r"Absolute residual",
    "rel_res":                       r"Relative residual",
    "train_rmse":                    r"Train RMSE",
    "train_posterior_samples_nll":   r"NLL (train samples)",
    "train_posterior_samples_mean":  r"Train posterior mean",
    "train_posterior_samples_var":   r"Train posterior variance",
    "train_mse":                      r"Train MSE",
    "train_r2":                       r"Train R²",
    "train_posterior_samples_mean_nll": r"NLL (train posterior mean)",
    "train_posterior_samples_var_nll":  r"NLL (train posterior variance)",
    "test_rmse":                     r"Test RMSE",
    "test_posterior_samples_nll":    r"NLL (posterior samples)",
    "test_posterior_samples_mean":   r"Posterior mean",
    "test_posterior_samples_var":    r"Posterior variance",
    "test_r2":                       r"Test R²",
    "test_mse":                      r"Test MSE",
    "test_posterior_samples_mean_nll": r"NLL (posterior mean)",
    "test_posterior_samples_var_nll":  r"NLL (posterior variance)"
}


OPT_LABELS = {
    "sdd": "SDD",
    "sap": "SAP",
    "nsap": "NSAP",
    "eigenpro2": "EigenPro 2.0",
    "eigenpro3": "EigenPro 3.0",
    "pcg": "PCG",
    "falkon": "Falkon",
    "mimosa": r"\texttt{Mimosa}",
}

RANK_LABEL = "r"
BLKSZ_LABEL = "b"
RHO_LABEL = r"\rho"
PRECOND_LABELS = {
    "nystrom": [r"Nystr$\ddot{\mathrm{o}}$m"],
    "partial_cholesky": [],
}
MODE_LABELS = {
    "greedy": "GC",
    "rpc": "RPC",
}
RHO_LABELS = {
    "damped": r"\mathrm{damped}",
    "regularization": r"\mathrm{regularization}",
}
SAMPLING_LABELS = {
    "uniform": "uniform",
    "rls": "RLS",
}

X_AXIS_LABELS: dict[str, str] = dict(
    time="Time (s)",
    datapasses="Full data passes",
    iters="Iterations",
)

NAN_REPLACEMENT: float = np.inf

X_AXIS_TIME_GRACE = 1.02

PERFORMANCE_AXIS_LABELS = {
    "x": "Fraction of time budget",
    "y": "Fraction of problems solved",
}

SORT_KEYS = ["opt", "accelerated", "sampling_method", "precond_type", "r", "b", "m"]
