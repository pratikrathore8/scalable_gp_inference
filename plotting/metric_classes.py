from warnings import warn

from dataclasses import dataclass
import numpy as np

from plotting.constants import X_AXIS_NAME_MAP


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

    @staticmethod
    def compute_statistics(
        metrics_list: list["MetricData"],
    ) -> tuple["MetricData", "MetricData", "MetricData"]:
        """
        Compute mean, minimum, and maximum for a list of MetricData objects.

        Args:
            metrics_list: List of MetricData objects

        Returns:
            Tuple of (mean_data, min_data, max_data)
        """
        # Check if all metrics have the same metric name
        reference = metrics_list[0]
        all_same_name = all(
            m.metric_name == reference.metric_name for m in metrics_list
        )
        if not all_same_name:
            raise ValueError("All MetricData objects must have the same metric name")

        # Check if all metrics have the same steps
        all_same_steps = all(
            np.array_equal(m.steps, reference.steps) for m in metrics_list
        )

        if not all_same_steps:
            warn(
                "Not all MetricData objects have the same steps. "
                "This may lead to incorrect results. "
                "This is likely because some runs were not finished. "
                "We will return None for the mean, min, and max data."
            )
            # Return None for mean, min, and max data
            return None, None, None

        # Stack metric data along a new axis
        stacked_metrics = np.stack([m.metric_data for m in metrics_list], axis=0)

        # Stack cum_times data
        stacked_cum_times = np.stack([m.cum_times for m in metrics_list], axis=0)

        # Compute means
        mean_metrics = np.mean(stacked_metrics, axis=0)
        mean_cum_times = np.mean(stacked_cum_times, axis=0)

        # Find actual min and max values
        min_metrics = np.min(stacked_metrics, axis=0)
        max_metrics = np.max(stacked_metrics, axis=0)

        # Check if all runs are finished
        all_finished = all(m.finished for m in metrics_list)

        # Create MetricData objects for mean, min, and max
        mean_data = MetricData(
            metric_data=mean_metrics,
            steps=reference.steps,
            datapasses=reference.datapasses,
            cum_times=mean_cum_times,
            finished=all_finished,
            metric_name=reference.metric_name,
        )

        min_data = MetricData(
            metric_data=min_metrics,
            steps=reference.steps,
            datapasses=reference.datapasses,
            cum_times=mean_cum_times,  # Using the same mean cum_times for all
            finished=all_finished,
            metric_name=reference.metric_name,
        )

        max_data = MetricData(
            metric_data=max_metrics,
            steps=reference.steps,
            datapasses=reference.datapasses,
            cum_times=mean_cum_times,  # Using the same mean cum_times for all
            finished=all_finished,
            metric_name=reference.metric_name,
        )

        return mean_data, min_data, max_data


@dataclass(kw_only=True, frozen=True)
class MetricDataBO(MetricData):
    """Data class to hold metric data for a BO run."""

    num_acquisitions: np.ndarray

    def get_plotting_x_axis(self, xaxis) -> str:
        if xaxis not in X_AXIS_NAME_MAP:
            raise ValueError(
                f"Invalid x-axis name: {xaxis}. Must be one of {X_AXIS_NAME_MAP}."
            )
        if xaxis == "datapasses":
            raise ValueError(
                "Datapasses is not available for Bayesian optimization runs."
            )
        elif xaxis == "iterations":
            raise ValueError(
                "Iterations is not available for Bayesian optimization runs."
            )
        elif xaxis == "time":
            return self.cum_times
        elif xaxis == "num_acquisitions":
            return self.num_acquisitions

    @staticmethod
    def compute_statistics(
        metrics_list: list["MetricDataBO"],
    ) -> tuple["MetricDataBO", "MetricDataBO", "MetricDataBO"]:
        """
        Compute mean, minimum, and maximum for a list of MetricDataBO objects.

        Args:
            metrics_list: List of MetricDataBO objects

        Returns:
            Tuple of (mean_data, min_data, max_data)
        """
        # Check if all metrics have the same metric name
        reference = metrics_list[0]
        all_same_name = all(
            m.metric_name == reference.metric_name for m in metrics_list
        )
        if not all_same_name:
            raise ValueError("All MetricDataBO objects must have the same metric name")

        # Check if all metrics have the same steps
        all_same_steps = all(
            np.array_equal(m.steps, reference.steps) for m in metrics_list
        )

        # Check if all metrics have the same num_acquisitions
        all_same_acquisitions = all(
            np.array_equal(m.num_acquisitions, reference.num_acquisitions)
            for m in metrics_list
        )

        if not all_same_steps or not all_same_acquisitions:
            warn(
                "Not all MetricDataBO objects have the same steps or acquisitions. "
                "This may lead to incorrect results. "
                "This is likely because some runs were not finished. "
                "We will return None for the mean, min, and max data."
            )
            # Return None for mean, min, and max data
            return None, None, None

        # Stack metric data along a new axis
        stacked_metrics = np.stack([m.metric_data for m in metrics_list], axis=0)

        # Stack cum_times data
        stacked_cum_times = np.stack([m.cum_times for m in metrics_list], axis=0)

        # Compute means
        mean_metrics = np.mean(stacked_metrics, axis=0)
        mean_cum_times = np.mean(stacked_cum_times, axis=0)

        # Find actual min and max values
        min_metrics = np.min(stacked_metrics, axis=0)
        max_metrics = np.max(stacked_metrics, axis=0)

        # Check if all runs are finished
        all_finished = all(m.finished for m in metrics_list)

        # Create MetricDataBO objects for mean, min, and max
        mean_data = MetricDataBO(
            metric_data=mean_metrics,
            steps=reference.steps,
            datapasses=reference.datapasses,
            cum_times=mean_cum_times,
            finished=all_finished,
            metric_name=reference.metric_name,
            num_acquisitions=reference.num_acquisitions,
        )

        min_data = MetricDataBO(
            metric_data=min_metrics,
            steps=reference.steps,
            datapasses=reference.datapasses,
            cum_times=mean_cum_times,  # Using the same mean cum_times for all
            finished=all_finished,
            metric_name=reference.metric_name,
            num_acquisitions=reference.num_acquisitions,
        )

        max_data = MetricDataBO(
            metric_data=max_metrics,
            steps=reference.steps,
            datapasses=reference.datapasses,
            cum_times=mean_cum_times,  # Using the same mean cum_times for all
            finished=all_finished,
            metric_name=reference.metric_name,
            num_acquisitions=reference.num_acquisitions,
        )

        return mean_data, min_data, max_data
