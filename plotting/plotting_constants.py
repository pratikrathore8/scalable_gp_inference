import numpy as np

USE_LATEX: bool = True
FONTSIZE:  int = 14
EXTENSION: str = "pdf"
BASE_SAVE_DIR: str = "./plots"


HPARAMS_TO_LABEL = {
    "sdd": ["precond", "r", "sampling_method"],
    "sap": ["b"],
    "pcg": ["precond", "r"],
}

SZ_COL: float = 8.0
SZ_ROW: float = 6.0


LEGEND_SPECS = {
    "loc": "lower center",
    "bbox_to_anchor": (0.5, -0.25),
    "ncol": 4,
    "frameon": False,
    "fontsize": FONTSIZE * 0.7
}

OPT_COLORS = {
    "sdd": "#8B0000",
    "sap": "#FF4500",
    "sap_nystrom": "#FF4500",
    "sap_identity": "tab:pink",
    "pcg": "#4169E1",
}

PRECOND_MARKERS = {
    "sap": {
        "nystrom": "o",
        "identity": "x",
    },
    "nystrom": {"damped": "+", "regularization": "8"},
}

SAMPLING_LINESTYLES = {
    "uniform": "solid",
    "rls": "dashed",
}

TOT_MARKERS: int = 4
MARKERSIZE: int = 4
ERRORBAND_ALPHA = 0.2

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
    "pcg": "PCG",
}

RANK_LABEL = "r"
BLKSZ_LABEL = "b"
RHO_LABEL = r"\rho"
PRECOND_LABELS = {
    "nystrom": [r"Nystr$\ddot{\mathrm{o}}$m"],
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


SORT_KEYS = ["opt", "accelerated", "sampling_method", "precond_type", "r", "b", "m"]


#------------------------------------------------------------
# Plotting constants for the Bayesian optimization task
#------------------------------------------------------------

BAYESIAN_OPT_BASE_SAVE_DIR: str = "./plots/bayesian_optimization"

BAYESIAN_OPT_METRIC_AX_PLOT_FNS: dict[str, str] = {
    "fn_max":                       "semilogy",
}

BAYESIAN_OPT_METRIC_LABELS: dict[str, str] = {
    "fn_max":                       r"Best objective value",
}


BAYESIAN_OPT_X_AXIS_LABELS: dict[str, str] = dict(
    time="Time (s)",
    num_acquisitions="Number of acquisitions",
    iters="Iterations",
)
