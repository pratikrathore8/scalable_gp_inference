from pathlib import Path
from typing import Dict, List, Optional, Sequence
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import warnings, shutil

from wandb_constants import CONFIG_KEYS, HPARAM_LABELS, BAYESIAN_OPT_CONFIG_KEYS
from plotting_constants import (
    USE_LATEX,
    FONTSIZE,
    EXTENSION,
    OPT_COLORS,
    OPT_LABELS,
    METRIC_LABELS,
    METRIC_AX_PLOT_FNS,
    LEGEND_SPECS,
    SZ_COL,
    SZ_ROW,
    X_AXIS_LABELS,
    ERRORBAND_ALPHA,
    PRECOND_MARKERS,
    TOT_MARKERS,
    MARKERSIZE,
    BAYESIAN_OPT_METRIC_AX_PLOT_FNS,
    BAYESIAN_OPT_METRIC_LABELS,
    BAYESIAN_OPT_X_AXIS_LABELS,
)

if USE_LATEX and not shutil.which("latex"):
    warnings.warn("LaTeX not found – falling back to Matplotlib's mathtext.")
    USE_LATEX = False

rcParams.update(
    {
        "font.size": FONTSIZE,
        "text.usetex": USE_LATEX,
        "axes.labelsize": FONTSIZE,
        "legend.fontsize": FONTSIZE * 0.8,
    }
)


class Plotter:
    def __init__(
        self,
        runs_data: Dict[str, pd.DataFrame],
        aggregated: bool,
        is_bayesian_opt: bool,
    ):
        self.runs_data = runs_data
        self.aggregated = aggregated
        self.is_bayesian_opt = is_bayesian_opt
        if not self.aggregated:
            self._validate_data()

    def _validate_data(self):
        """Ensure required config keys exist in all DataFrames"""
        if self.is_bayesian_opt:
            required_columns = {"x_value", "run_id", "run_name", CONFIG_KEYS["SOLVER"]}
        else:
            required_columns = {
                "x_value",
                "run_id",
                "run_name",
                CONFIG_KEYS["SOLVER"],
                CONFIG_KEYS["DATASET"],
            }

        for run_id, df in self.runs_data.items():
            missing = required_columns - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns {missing} in run {run_id}")

    def _get_solver_config(self, df: pd.DataFrame) -> str:
        """Get solver config with fallback handling"""
        try:
            solver = (
                df[CONFIG_KEYS["SOLVER"]].iloc[0]
                if not self.is_bayesian_opt
                else df[BAYESIAN_OPT_CONFIG_KEYS["SOLVER"]].iloc[0]
            )
        except KeyError:
            raise ValueError(
                f"Missing solver config in run {df['run_id'].iloc[0]}. "
                f"Available columns: {df.columns.tolist()}"
            )

        color_key = (
            df[CONFIG_KEYS["SOLVER"]].iloc[0]
            if not self.is_bayesian_opt
            else df[BAYESIAN_OPT_CONFIG_KEYS["SOLVER"]].iloc[0]
        )
        precond_type = (
            df.get("sap_precond", pd.Series([""])).iloc[0].lower()
            if color_key == "sap"
            else ""
        )

        if color_key == "sap" and precond_type:
            color = OPT_COLORS[f"sap_{precond_type}"]
            marker = PRECOND_MARKERS["sap"][precond_type]
        else:
            color = OPT_COLORS.get(color_key, "k")
            marker = None

        hparams = HPARAM_LABELS.get(solver, [])
        parts = [OPT_LABELS.get(solver, solver)]

        if solver == "sap" and "sap_precond" in df.columns:
            parts.append(f"({df['sap_precond'].iloc[0]})")

        for hp in hparams:
            if hp in df.columns:
                parts.append(f"{hp}={df[hp].iloc[0]}")

        return {"label": " ".join(parts), "color": color, "marker": marker}

    def _name_y_axis(self, metric_path: str) -> str:
        """Convert metric path to clean label"""
        metric_key = metric_path.split(".")[-1]
        y_axis_label = (
            METRIC_LABELS.get(metric_key, metric_key.replace("_", " ").title())
            if not self.is_bayesian_opt
            else BAYESIAN_OPT_METRIC_LABELS.get(
                metric_key, metric_key.replace("_", " ").title()
            )
        )
        return y_axis_label

    def _name_dataset(self, dataset_path: str) -> str:
        """Clean dataset name formatting"""
        return dataset_path.replace("_", " ").title()

    def plot_errorbars(
        self,
        metric: str,
        title: str | None = None,
        save_path: Path | None = None,
        capsize: float = 4.0,
    ):

        fig, ax = plt.subplots(figsize=(SZ_COL, SZ_ROW))

        labels, means, errs, colors = [], [], [], []

        for raw_key, df in self.runs_data.items():
            means.append(df[f"{metric}_mean"].iloc[0])
            errs.append(df[f"{metric}_std"].iloc[0])

            raw_key = raw_key.strip()
            solver_key = raw_key.split()[0].lower()
            variant = raw_key[len(solver_key) :].strip()

            base_label = OPT_LABELS.get(solver_key, solver_key.upper())
            label = f"{base_label} {variant}".strip()
            labels.append(label)

            if solver_key == "sap":
                if "identity" in variant.lower():
                    color_key = "sap_identity"
                elif "nystrom" in variant.lower():
                    color_key = "sap_nystrom"
                else:
                    color_key = "sap"
            else:
                color_key = solver_key
            colors.append(OPT_COLORS.get(color_key, "k"))

        x = np.arange(len(labels))
        for xi, m, err, col in zip(x, means, errs, colors):
            ax.errorbar(
                xi,
                m,
                yerr=err,
                fmt="s",
                capsize=capsize,
                linestyle="none",
                color=col,
                ecolor=col,
                markeredgecolor=col,
                markerfacecolor=col,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(self._name_y_axis(metric))

        if title:
            ax.set_title(title)

        self._save_or_show(fig, save_path)
        return fig, ax

    def plot_single_metric(
        self,
        y_metric: str,
        x_axis: str,
        log_y: bool,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ) -> tuple[Figure, Axes]:
        """Plot single metric comparison across all runs."""
        fig, ax = plt.subplots(figsize=(SZ_COL, SZ_ROW))

        for run_id, df in self.runs_data.items():
            config = self._get_solver_config(df)

            plot_fn = getattr(ax, METRIC_AX_PLOT_FNS.get(y_metric, "plot"))
            plot_fn(
                df["x_value"],
                df[y_metric],
                label=config["label"],
                color=config["color"],
                marker=config["marker"],
                markersize=MARKERSIZE,
                markevery=len(df["x_value"]) // TOT_MARKERS,
            )

        x_axis_label = (
            X_AXIS_LABELS.get(x_axis, x_axis)
            if not self.is_bayesian_opt
            else BAYESIAN_OPT_X_AXIS_LABELS.get(x_axis, x_axis)
        )

        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(self._name_y_axis(y_metric))
        ax.legend(**LEGEND_SPECS)

        if log_y:
            ax.set_yscale("log")

        if title is None and not self.is_bayesian_opt:
            dataset = self._name_dataset(df[CONFIG_KEYS["DATASET"]].iloc[0])
            title = f"{dataset}"
        elif title is None and self.is_bayesian_opt:
            title = "Bayesian optimization task"

        if title:
            ax.set_title(title, fontsize=FONTSIZE * 0.9)

        self._save_or_show(fig, save_path)
        return fig, ax

    def plot_metric_grid(
        self,
        y_metrics: Sequence[str],
        x_axis: str,
        log_y: bool = True,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ) -> tuple[Figure, List[Axes]]:
        """Plot grid of metrics with consistent x-axis."""
        n_metrics = len(y_metrics)
        n_cols = int(np.ceil(np.sqrt(n_metrics)))
        n_rows = int(np.ceil(n_metrics / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(SZ_COL * n_cols, SZ_ROW * n_rows), squeeze=False
        )
        axes = axes.flatten()

        for idx, y_metric in enumerate(y_metrics):
            ax = axes[idx]
            for run_id, df in self.runs_data.items():
                config = self._get_solver_config(df)
                solver = (
                    df[CONFIG_KEYS["SOLVER"]].iloc[0]
                    if not self.is_bayesian_opt
                    else df[BAYESIAN_OPT_CONFIG_KEYS["SOLVER"]].iloc[0]
                )
                color = OPT_COLORS.get(solver, "k")

                plot_fn = (
                    getattr(ax, METRIC_AX_PLOT_FNS.get(y_metric, "plot"))
                    if not self.is_bayesian_opt
                    else getattr(
                        ax, BAYESIAN_OPT_METRIC_AX_PLOT_FNS.get(y_metric, "plot")
                    )
                )

                plot_fn(
                    df["x_value"],
                    df[y_metric],
                    label=config["label"],
                    color=config["color"],
                    marker=config["marker"],
                    markersize=MARKERSIZE,
                    markevery=len(df["x_value"]) // TOT_MARKERS,
                )

            x_axis_label = (
                X_AXIS_LABELS.get(x_axis, x_axis)
                if not self.is_bayesian_opt
                else BAYESIAN_OPT_X_AXIS_LABELS.get(x_axis, x_axis)
            )
            ax.set_xlabel(x_axis_label)
            ax.set_ylabel(self._name_y_axis(y_metric))
            ax.legend(**LEGEND_SPECS)

            if log_y:
                ax.set_yscale("log")

            if title:
                ax.set_title(title)

        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        self._save_or_show(fig, save_path)
        return fig, axes

    def _save_or_show(self, fig: Figure, save_path: Optional[Path] = None):
        """Handle figure output."""
        fig.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path.with_suffix(f".{EXTENSION}"), bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def plot_with_errorbands(
        self,
        aggregated_data: Dict[str, Dict[str, pd.DataFrame]],
        y_metric: str,
        x_axis: str,
        log_y: bool,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ) -> tuple[Figure, Axes]:
        """Plot single metric with error bands across all runs."""
        fig, ax = plt.subplots(figsize=(SZ_COL, SZ_ROW))

        for group_key, data in aggregated_data.items():
            mean_df = data["mean"]
            std_df = data["std"]
            num_runs = data["num_runs"]

            solver = (
                mean_df[CONFIG_KEYS["SOLVER"]].iloc[0]
                if not self.is_bayesian_opt
                else mean_df[BAYESIAN_OPT_CONFIG_KEYS["SOLVER"]].iloc[0]
            )
            config = self._get_solver_config(mean_df)

            plot_fn = (
                getattr(ax, METRIC_AX_PLOT_FNS.get(y_metric, "plot"))
                if not self.is_bayesian_opt
                else getattr(ax, BAYESIAN_OPT_METRIC_AX_PLOT_FNS.get(y_metric, "plot"))
            )

            plot_fn(
                mean_df["x_value"],
                mean_df[y_metric],
                label=config["label"],
                color=config["color"],
                marker=config["marker"],
                markersize=MARKERSIZE,
                markevery=len(mean_df["x_value"]) // TOT_MARKERS,
            )

            ax.fill_between(
                mean_df["x_value"],
                mean_df[y_metric] - std_df[y_metric],
                mean_df[y_metric] + std_df[y_metric],
                color=config["color"],
                alpha=ERRORBAND_ALPHA,
            )

        x_axis_label = (
            X_AXIS_LABELS.get(x_axis, x_axis)
            if not self.is_bayesian_opt
            else BAYESIAN_OPT_X_AXIS_LABELS.get(x_axis, x_axis)
        )
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(self._name_y_axis(y_metric))
        ax.legend(**LEGEND_SPECS)

        if log_y:
            ax.set_yscale("log")

        if title:
            ax.set_title(title, fontsize=FONTSIZE * 0.9)

        self._save_or_show(fig, save_path)
        return fig, ax

    def plot_metric_grid_with_errorbands(
        self,
        aggregated_data: Dict[str, Dict[str, pd.DataFrame]],
        y_metrics: Sequence[str],
        x_axis: str,
        log_y: bool = True,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
    ) -> tuple[Figure, List[Axes]]:
        """Plot grid of metrics with error bands."""
        n_metrics = len(y_metrics)
        n_cols = int(np.ceil(np.sqrt(n_metrics)))
        n_rows = int(np.ceil(n_metrics / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(SZ_COL * n_cols, SZ_ROW * n_rows), squeeze=False
        )
        axes = axes.flatten()

        handles_list = []
        labels_list = []

        for idx, y_metric in enumerate(y_metrics):
            ax = axes[idx]

            for group_key, data in aggregated_data.items():
                mean_df = data["mean"]
                std_df = data["std"]
                num_runs = data["num_runs"]

                solver = (
                    mean_df[CONFIG_KEYS["SOLVER"]].iloc[0]
                    if not self.is_bayesian_opt
                    else mean_df[BAYESIAN_OPT_CONFIG_KEYS["SOLVER"]].iloc[0]
                )
                config = self._get_solver_config(mean_df)

                plot_fn = (
                    getattr(ax, METRIC_AX_PLOT_FNS.get(y_metric, "plot"))
                    if not self.is_bayesian_opt
                    else getattr(
                        ax, BAYESIAN_OPT_METRIC_AX_PLOT_FNS.get(y_metric, "plot")
                    )
                )

                line = plot_fn(
                    mean_df["x_value"],
                    mean_df[y_metric],
                    label=config["label"],
                    color=config["color"],
                    marker=config["marker"],
                    markersize=MARKERSIZE,
                    markevery=len(mean_df["x_value"]) // TOT_MARKERS,
                )

                ax.fill_between(
                    mean_df["x_value"],
                    mean_df[y_metric] - std_df[y_metric],
                    mean_df[y_metric] + std_df[y_metric],
                    color=config["color"],
                    alpha=ERRORBAND_ALPHA,
                )

                if idx == 0:
                    handles_list.append(line[0])
                    labels_list.append(config["label"])

            x_axis_label = (
                X_AXIS_LABELS.get(x_axis, x_axis)
                if not self.is_bayesian_opt
                else BAYESIAN_OPT_X_AXIS_LABELS.get(x_axis, x_axis)
            )
            ax.set_xlabel(x_axis_label)
            ax.set_ylabel(self._name_y_axis(y_metric))

            if log_y:
                ax.set_yscale("log")

            if title:
                ax.set_title(title)

        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        legend_specs = LEGEND_SPECS.copy()
        legend_specs["ncol"] = len(handles_list)

        legend_position = {
            "loc": "lower center",
            "bbox_to_anchor": (0.5, -0.05),
            "ncol": legend_specs["ncol"],
            "frameon": legend_specs.get("frameon", False),
            "fontsize": legend_specs.get("fontsize", FONTSIZE * 0.7),
        }

        fig.legend(handles_list, labels_list, **legend_position)

        self._save_or_show(fig, save_path)
        return fig, axes
