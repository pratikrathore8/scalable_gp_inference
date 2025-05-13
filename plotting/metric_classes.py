from dataclasses import dataclass
import numpy as np

from plotting.constants import X_AXIS_NAME_MAP, METRIC_NAME_BASE, METRIC_NAME_MAP


@dataclass(kw_only=True, frozen=True)
class MetricData:
    """Data class to hold metric data for a run."""

    metric_name: str
    metric_data: np.ndarray
    steps: np.ndarray
    datapasses: np.ndarray
    cum_times: np.ndarray
    finished: bool

    def get_final_time(self) -> float:
        """Return the last element in the cum_times array."""
        if len(self.cum_times) == 0:
            raise ValueError("cum_times array is empty.")
        return self.cum_times[-1]

    def get_plotting_name(self) -> str:
        """Return the name of the metric for plotting."""
        return self.metric_name

    def get_plotting_x_axis(self, xaxis) -> str:
        if xaxis not in X_AXIS_NAME_MAP:
            raise ValueError(
                f"Invalid x-axis name: {xaxis}. Must be one of {X_AXIS_NAME_MAP}."
            )
        if xaxis == "datapasses":
            return self.datapasses
        elif xaxis == "iterations":
            return self.steps
        elif xaxis == "time":
            return self.cum_times


class WandbRun:
    def __init__(self, run):
        self.run = run

    @property
    def opt_name(self) -> str:
        if self.run.config["solver_name"] == "sap":
            if self.run.config["solver_config"].get("precond_config", None):
                return r"\texttt{ADASAP}"
            else:
                return r"\texttt{ADASAP-I}"
        elif self.run.config["solver_name"] == "sdd":
            return f"SDD-{self.run.config['opt_step_size_unscaled']}"
        elif self.run.config["solver_name"] == "pcg":
            if self.run.config["solver_config"].get("precond_config", None):
                return "PCG"
        else:
            raise ValueError(f"Unknown solver name: {self.run.config['solver_name']}")

    @property
    def color(self) -> str:
        if self.opt_name == r"\texttt{ADASAP}":
            return "tab:blue"
        elif self.opt_name == r"\texttt{ADASAP-I}":
            return "tab:orange"
        elif self.opt_name == "SDD-1":
            return "tab:purple"
        elif self.opt_name == "SDD-10":
            return "tab:pink"
        elif self.opt_name == "SDD-100":
            return "tab:brown"
        elif self.opt_name == "PCG":
            return "tab:olive"
        else:
            raise ValueError(f"Unknown optimizer name: {self.opt_name}")

    def _get_num_blocks(self) -> int:
        if self.run.config["solver_name"] in ["sap", "sdd"]:
            return self.run.config["opt_num_blocks"]
        elif self.run.config["solver_name"] == "pcg":
            return 1

    def get_metric_data(self, metric: str) -> MetricData:
        full_metric_name = f"{METRIC_NAME_BASE}{metric}"
        run_hist = self.run.scan_history(keys=[full_metric_name, "_step", "iter_time"])

        # Extract raw data
        metric_data = np.array([x[full_metric_name] for x in run_hist])
        steps = np.array([x["_step"] for x in run_hist])
        times = np.array([x["iter_time"] for x in run_hist])

        # Identify unique step indices -- this is needed to remove duplicates
        _, unique_indices = np.unique(steps, return_index=True)
        # Sort to maintain original order
        unique_indices = np.sort(unique_indices)

        # Filter to keep only unique step entries
        metric_data = metric_data[unique_indices]
        times = times[unique_indices]
        steps = steps[unique_indices]

        num_blocks = self._get_num_blocks()
        datapasses = steps / num_blocks

        # Calculate cumulative times
        cum_times = np.cumsum(times)

        return MetricData(
            metric_name=METRIC_NAME_MAP[metric],
            metric_data=metric_data,
            steps=steps,
            datapasses=datapasses,
            cum_times=cum_times,
            finished=True if self.run.state == "finished" else False,
        )
