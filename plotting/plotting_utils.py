from pathlib import Path
from typing import Dict, List, Optional, Sequence
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from wandb_constants import (
    METRIC_PATHS,
    CONFIG_KEYS,
    X_AXIS_OPTIONS,
    SOLVERS,
    DATASETS,
    HPARAM_LABELS,
)
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
    NAN_REPLACEMENT,
)

rcParams.update({
    "font.size": FONTSIZE,
    "text.usetex": USE_LATEX,
    "axes.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE * 0.8,
})


class Plotter:
    def __init__(self, runs_data: Dict[str, pd.DataFrame]):
        self.runs_data = runs_data
        self._validate_data()

    def _validate_data(self):
        """Ensure required config keys exist in all DataFrames"""
        required_columns = {
            "x_value", "run_id", "run_name",
            CONFIG_KEYS["SOLVER"], CONFIG_KEYS["DATASET"]
        }

        for run_id, df in self.runs_data.items():
            missing = required_columns - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns {missing} in run {run_id}")

    def _get_solver_config(self, df: pd.DataFrame) -> str:
        """Get solver config with fallback handling"""
        try:
            solver = df[CONFIG_KEYS["SOLVER"]].iloc[0]
        except KeyError:
            raise ValueError(
                f"Missing solver config in run {df['run_id'].iloc[0]}. "
                f"Available columns: {df.columns.tolist()}"
            )

        hparams = HPARAM_LABELS.get(solver, [])
        parts = [OPT_LABELS.get(solver, solver)]

        for hp in hparams:
            if hp in df.columns:
                parts.append(f"{hp}={df[hp].iloc[0]}")

        return " ".join(parts)

    def _name_y_axis(self, metric_path: str) -> str:
        """Convert metric path to clean label"""
        metric_key = metric_path.split('.')[-1]
        return METRIC_LABELS.get(metric_key,
                                 metric_key.replace('_', ' ').title())

    def _name_dataset(self, dataset_path: str) -> str:
        """Clean dataset name formatting"""
        return dataset_path.replace("_", " ").title()

    def plot_single_metric(
            self,
            y_metric: str,
            x_axis: str = X_AXIS_OPTIONS["TIME"],
            log_y: bool = True,
            title: Optional[str] = None,
            save_path: Optional[Path] = None,
    ) -> tuple[Figure, Axes]:
        """Plot single metric comparison across all runs."""
        fig, ax = plt.subplots(figsize=(SZ_COL, SZ_ROW))

        for run_id, df in self.runs_data.items():
            solver_label = self._get_solver_config(df)
            color = OPT_COLORS.get(df[CONFIG_KEYS["SOLVER"]].iloc[0], "k")

            plot_fn = getattr(ax, METRIC_AX_PLOT_FNS.get(y_metric, "plot"))
            plot_fn(df["x_value"], df[y_metric], label=solver_label,
                    color=color)

        ax.set_xlabel(X_AXIS_LABELS.get(x_axis, x_axis))
        ax.set_ylabel(self._name_y_axis(y_metric))
        ax.legend(**LEGEND_SPECS)

        if log_y:
            ax.set_yscale("log")

        # if title is None:
        #     dataset = self._name_dataset(df[CONFIG_KEYS["DATASET"]].iloc[0])
        #     title = f"{dataset}"
        # ax.set_title(title, fontsize=FONTSIZE*0.9)

        self._save_or_show(fig, save_path)
        return fig, ax

    def plot_metric_grid(
            self,
            y_metrics: Sequence[str],
            x_axis: str = X_AXIS_OPTIONS["TIME"],
            log_y: bool = True,
            title: Optional[str] = None,
            save_path: Optional[Path] = None,
    ) -> tuple[Figure, List[Axes]]:
        """Plot grid of metrics with consistent x-axis."""
        n_metrics = len(y_metrics)
        n_cols = int(np.ceil(np.sqrt(n_metrics)))
        n_rows = int(np.ceil(n_metrics / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(SZ_COL * n_cols, SZ_ROW * n_rows),
            squeeze=False
        )
        axes = axes.flatten()

        for idx, y_metric in enumerate(y_metrics):
            ax = axes[idx]
            for run_id, df in self.runs_data.items():
                solver_label = self._get_solver_config(df)
                color = OPT_COLORS.get(df[CONFIG_KEYS["SOLVER"]].iloc[0], "k")

                plot_fn = getattr(ax, METRIC_AX_PLOT_FNS.get(y_metric, "plot"))
                plot_fn(df["x_value"], df[y_metric], label=solver_label,
                        color=color)

            ax.set_xlabel(X_AXIS_LABELS.get(x_axis, x_axis))
            ax.set_ylabel(self._name_y_axis(y_metric))
            ax.legend(**LEGEND_SPECS)

            if log_y:
                ax.set_yscale("log")

            # ax.set_title(METRIC_LABELS.get(y_metric, y_metric))

        # Hide unused axes
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        # if title:
        #     fig.suptitle(title, y=1.02, fontsize=FONTSIZE)

        self._save_or_show(fig, save_path)
        return fig, axes

    def _save_or_show(self, fig: Figure, save_path: Optional[Path] = None):
        """Handle figure output."""
        fig.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path.with_suffix(f".{EXTENSION}"),
                        bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


# Example Usage ----------------------------------------------------------------
if __name__ == "__main__":
    from wandb_utils import get_project_runs, filter_runs, organize_runs_data

    runs = get_project_runs("sketchy-opts", "gp_inference_acsincome")
    print(f"Total runs found: {len(runs)}")
    filtered = filter_runs(runs,
                           require_all={
                               CONFIG_KEYS["DATASET"]: "acsincome",
                               CONFIG_KEYS["SOLVER"]: ["sap", "pcg"]
                           })
    print(f"Runs after filtering: {len(filtered)}")
    runs_data = organize_runs_data(filtered, [METRIC_PATHS["TEST_RMSE"],
                                              METRIC_PATHS["TEST_R2"]])

    plotter = Plotter(runs_data)

    plotter.plot_single_metric(
        y_metric=METRIC_PATHS["TEST_RMSE"],
        x_axis=X_AXIS_OPTIONS["TIME"],
        save_path=Path("plots/acsincome/rmse_comparison")
    )

    plotter.plot_metric_grid(
        y_metrics=[METRIC_PATHS["TEST_RMSE"], METRIC_PATHS["TEST_R2"]],
        save_path=Path("plots/acsincome/multi_plot_comparison")
    )