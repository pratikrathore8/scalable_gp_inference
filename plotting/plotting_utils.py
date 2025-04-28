import pickle
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from plotting_constants import (
    USE_LATEX,
    FONTSIZE,
    EXTENSION,
    BASE_SAVE_DIR,
    OPT_COLORS,
    OPT_LABELS,
    METRIC_LABELS,
    METRIC_AX_PLOT_FNS,
    LEGEND_SPECS,
    SZ_COL,
    SZ_ROW,
)

# LaTeX / fonts global switch
rcParams.update(
    {
        "font.size": FONTSIZE,
        "text.usetex": USE_LATEX,
        "axes.labelsize": FONTSIZE,
        "legend.fontsize": FONTSIZE * 0.8,
    }
)


def _latest_result_pkl(dataset: str, solver: str, results_dir: str | Path = ".results") -> Path:
    """Return the most recently modified result file matching *dataset*/*solver*."""
    results_dir = Path(results_dir)
    pattern = f"{dataset}_{solver}_*.pkl"
    matches = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"No result matching pattern {pattern!r} inside {results_dir.resolve()}"
        )
    return matches[0]


def get_results(
    dataset_name: str,
    solver: str,
    *,
    results_dir: str | Path = ".results",
) -> tuple[Mapping[int, Mapping[str, Any]], np.ndarray]:
    """
    Load *log* and *W_star* from the newest experiment file for `dataset_name`/`solver`.

    Returns
    -------
    log : dict[int, dict]
        Dict keyed by iteration with the same structure emitted by `KernelLinSys.solve`.
    W_star : np.ndarray
        The solution weights.
    """
    pkl_path = _latest_result_pkl(dataset_name, solver, results_dir)
    with open(pkl_path, "rb") as fh:
        payload = pickle.load(fh)

    result_dict = payload["result"]
    return result_dict["log"], result_dict["W_star"]


def _maybe_item(x):
    """Return a Python scalar if *x* is a 0-D or 1-element tensor/array."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            if x.ndim == 0 or x.numel() == 1:
                return x.item()
            return x.cpu().numpy()
    except ModuleNotFoundError:
        pass
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.squeeze())
        return x

    return x



def log_to_dataframe(log: Mapping[int, Mapping[str, Any]]) -> pd.DataFrame:
    """
    Flatten the nested *log* dict into a tidy `pd.DataFrame`.

    Every scalar metric becomes a column; missing values are filled with NaN.
    Non-scalar tensors are promoted to NumPy arrays so they can be stringified
    downstream if needed.
    """
    records: list[dict[str, Any]] = []

    for it, entry in sorted(log.items()):
        rec: dict[str, Any] = {
            "iter": it,
            "cum_time": entry.get("cum_time"),
        }

        metrics = entry.get("metrics", {})
        cb_metrics = metrics.get("callback", {})
        int_metrics = metrics.get("internal_metrics", {})

        for d in (cb_metrics, int_metrics):
            for k, v in d.items():
                rec[k] = _maybe_item(v)

        records.append(rec)

    df = pd.DataFrame.from_records(records).sort_values("iter").reset_index(drop=True)
    print(df)
    return df



def _prep_ax(ax, *, log_y: bool) -> None:
    if log_y:
        ax.set_yscale("log")


def plot_single(
    df: pd.DataFrame,
    *,
    x_metric: str,
    y_metric: str,
    dataset_name: str,
    solver: str,
    log_y: bool = False,
    title: str | None = None,
    save: str | Path | None = None,
):

    color = OPT_COLORS.get(solver, None)
    label = OPT_LABELS.get(solver, solver)

    fig, ax = plt.subplots(figsize=(SZ_COL, SZ_ROW))
    plot_fn_name = METRIC_AX_PLOT_FNS.get(y_metric, "plot")
    plot_fn = getattr(ax, plot_fn_name)

    plot_fn(df[x_metric], df[y_metric], label=label, color=color)

    ax.set_xlabel(METRIC_LABELS.get(x_metric, x_metric))
    ax.set_ylabel(METRIC_LABELS.get(y_metric, y_metric))
    _prep_ax(ax, log_y=log_y)

    if title is None:
        title = f"{dataset_name} – {label}"
    ax.set_title(title)

    ax.legend(**LEGEND_SPECS)

    fig.tight_layout()
    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save.with_suffix(f".{EXTENSION}"), bbox_inches="tight")
    else:
        plt.show()

    return fig, ax


def plot_multiple(
    dfs: list[pd.DataFrame],
    *,
    x_metric: str,
    y_metric: str,
    dataset_name: str,
    solvers: list[str],
    num_rows: int,
    num_columns: int,
    log_y: bool = True,
    title: str | None = None,
    save: str | Path | None = None,
):

    if len(dfs) != len(solvers):
        raise ValueError("`dfs` and `solvers` must have the same length")

    tot_plots = len(dfs)
    if tot_plots > num_rows * num_columns:
        raise ValueError("Grid too small for number of plots requested")

    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(SZ_COL * num_columns, SZ_ROW * num_rows),
        squeeze=False,
    )

    for idx, (df, solver) in enumerate(zip(dfs, solvers)):
        r, c = divmod(idx, num_columns)
        ax = axes[r][c]
        color = OPT_COLORS.get(solver, None)
        label = OPT_LABELS.get(solver, solver)
        plot_fn_name = METRIC_AX_PLOT_FNS.get(y_metric, "plot")
        getattr(ax, plot_fn_name)(df[x_metric], df[y_metric], label=label, color=color)

        ax.set_xlabel(METRIC_LABELS.get(x_metric, x_metric))
        ax.set_ylabel(METRIC_LABELS.get(y_metric, y_metric))
        _prep_ax(ax, log_y=log_y)
        ax.legend(**LEGEND_SPECS)

        ax.set_title(label)

    # Hide unused sub-axes
    for idx in range(tot_plots, num_rows * num_columns):
        r, c = divmod(idx, num_columns)
        axes[r][c].set_visible(False)

    if title:
        fig.suptitle(title, y=1.02, fontsize=FONTSIZE)

    fig.tight_layout()
    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save.with_suffix(f".{EXTENSION}"), bbox_inches="tight")
    else:
        plt.show()

    return fig, axes


