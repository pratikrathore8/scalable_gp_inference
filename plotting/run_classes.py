import numpy as np

from plotting.constants import METRIC_NAME_BASE, METRIC_NAME_MAP
from plotting.metric_classes import MetricData, MetricDataBO


class WandbRun:
    def __init__(self, run):
        self.run = run

    @property
    def opt_name(self) -> str:
        config = self.run.config
        if config["solver_name"] == "sap":
            if config["solver_config"].get("precond_config", None):
                return r"\texttt{ADASAP}"
            else:
                return r"\texttt{ADASAP-I}"
        elif config["solver_name"] == "sdd":
            return f"SDD-{config['opt_step_size_unscaled']}"
        elif config["solver_name"] == "pcg":
            if config["solver_config"].get("precond_config", None):
                return "PCG"
        else:
            raise ValueError(f"Unknown solver name: {config['solver_name']}")

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
        config = self.run.config
        if config["solver_name"] in ["sap", "sdd"]:
            return config["opt_num_blocks"]
        elif config["solver_name"] == "pcg":
            return 1

    def get_metric_data(self, metric: str) -> MetricData:
        full_metric_name = METRIC_NAME_BASE + metric
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


class WandbRunBO(WandbRun):
    def __init__(self, run):
        super().__init__(run)

    @property
    def opt_name(self) -> str:
        config = self.run.config
        if config["ts_config"]["acquisition_method"] == "random_search":
            return "Random Search"
        elif config["solver_name"] == "sdd":
            # NOTE(pratik): We forgot to add the unscaled step size to the config
            # in the SDD case. This is a workaround fix.
            step_size_unscaled = (
                config["bo_config"]["num_init_samples"]
                * config["solver_config"]["step_size"]
            )
            step_size_unscaled = round(step_size_unscaled)  # make it an int
            return f"SDD-{step_size_unscaled}"
        else:
            return super().opt_name

    @property
    def color(self) -> str:
        if self.opt_name == "Random Search":
            return "k"
        else:
            return super().color

    def get_metric_data(self, metric: str) -> MetricDataBO:
        run_hist = self.run.scan_history(
            keys=[metric, "_step", "iter_time", "num_acquisitions"]
        )

        # Extract raw data
        metric_data = np.array([x[metric] for x in run_hist])
        steps = np.array([x["_step"] for x in run_hist])
        times = np.array([x["iter_time"] for x in run_hist])
        num_acquisitions = np.array([x["num_acquisitions"] for x in run_hist])

        # Identify unique step indices -- this is needed to remove duplicates
        _, unique_indices = np.unique(steps, return_index=True)
        # Sort to maintain original order
        unique_indices = np.sort(unique_indices)

        # Filter to keep only unique step entries
        metric_data = metric_data[unique_indices]
        times = times[unique_indices]
        steps = steps[unique_indices]
        num_acquisitions = num_acquisitions[unique_indices]

        # Calculate cumulative times
        cum_times = np.cumsum(times)

        return MetricDataBO(
            metric_name=METRIC_NAME_MAP[metric],
            metric_data=metric_data,
            steps=steps,
            datapasses=None,
            cum_times=cum_times,
            num_acquisitions=num_acquisitions,
            finished=True if self.run.state == "finished" else False,
        )
