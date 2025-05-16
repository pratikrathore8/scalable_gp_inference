from warnings import warn
from dataclasses import dataclass
import numpy as np
from typing import TypeVar, Type

from plotting.constants import X_AXIS_NAME_MAP

# Define a type variable for self-referential type hints
T = TypeVar("T", bound="MetricData")


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

    @classmethod
    def compute_statistics(cls: Type[T], metrics_list: list[T]) -> tuple[T, T, T]:
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
            raise ValueError(
                f"All {cls.__name__} objects must have the same metric name"
            )

        # Get fields to check for equality
        fields_to_check = ["steps"]

        # Check required fields are equal across all metrics
        checks_passed = True
        for field in fields_to_check:
            is_equal = all(
                np.array_equal(getattr(m, field), getattr(reference, field))
                for m in metrics_list
            )
            if not is_equal:
                warn(
                    f"Not all {cls.__name__} objects have the same {field}. "
                    f"This may lead to incorrect results. "
                    f"This is likely because some runs were not finished."
                )
                checks_passed = False

        # Additional checks for subclasses
        subclass_checks_passed = cls._additional_consistency_checks(
            metrics_list, reference
        )

        if not checks_passed or not subclass_checks_passed:
            warn("Returning None for the mean, min, and max data.")
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

        # Build kwargs for creating new objects
        common_kwargs = {
            "metric_data": mean_metrics,
            "steps": reference.steps,
            "datapasses": reference.datapasses,
            "cum_times": mean_cum_times,
            "finished": all_finished,
            "metric_name": reference.metric_name,
        }

        # Add additional kwargs for subclasses
        additional_kwargs = cls._get_additional_kwargs(reference)
        kwargs = {**common_kwargs, **additional_kwargs}

        # Create objects for mean, min, and max
        mean_data = cls(**kwargs)

        # Update metric_data for min and max
        min_kwargs = {**kwargs, "metric_data": min_metrics}
        max_kwargs = {**kwargs, "metric_data": max_metrics}

        min_data = cls(**min_kwargs)
        max_data = cls(**max_kwargs)

        return mean_data, min_data, max_data

    @classmethod
    def _additional_consistency_checks(
        cls, metrics_list: list[T], reference: T
    ) -> bool:
        """
        Perform additional consistency checks specific to subclasses.

        Args:
            metrics_list: List of MetricData objects
            reference: Reference MetricData object

        Returns:
            True if all checks pass, False otherwise
        """
        # Base class has no additional checks
        return True

    @classmethod
    def _get_additional_kwargs(cls, reference: T) -> dict:
        """
        Get additional kwargs specific to subclasses.

        Args:
            reference: Reference MetricData object

        Returns:
            Dictionary of additional kwargs
        """
        # Base class has no additional kwargs
        return {}


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

    @classmethod
    def _additional_consistency_checks(
        cls, metrics_list: list["MetricDataBO"], reference: "MetricDataBO"
    ) -> bool:
        """
        Check if all MetricDataBO objects have the same num_acquisitions.

        Args:
            metrics_list: List of MetricDataBO objects
            reference: Reference MetricDataBO object

        Returns:
            True if all checks pass, False otherwise
        """
        all_same_acquisitions = all(
            np.array_equal(m.num_acquisitions, reference.num_acquisitions)
            for m in metrics_list
        )

        if not all_same_acquisitions:
            warn(
                "Not all MetricDataBO objects have the same acquisitions. "
                "This may lead to incorrect results."
            )
            return False

        return True

    @classmethod
    def _get_additional_kwargs(cls, reference: "MetricDataBO") -> dict:
        """
        Get additional kwargs for MetricDataBO.

        Args:
            reference: Reference MetricDataBO object

        Returns:
            Dictionary with num_acquisitions
        """
        return {"num_acquisitions": reference.num_acquisitions}
