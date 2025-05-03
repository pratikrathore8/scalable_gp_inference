from dataclasses import dataclass

import torch

from ..kernel_linsys import KernelLinSys  # noqa: F401
from ..random_features import RandomFeatures
from .configs import BayesOptConfig, TSConfig  # noqa: F401


@dataclass(kwargs_only=True, frozen=False)
class ThompsonDataset:
    """
    Represents a dataset for Thompson sampling.

    Attributes:
        x (torch.Tensor): The input data.
        y (torch.Tensor): The target data.
        N (int): The number of data points.
        D (int): The number of features.
    """

    x: torch.Tensor
    y: torch.Tensor
    N: int
    D: int


# NOTE(pratik): we freeze the parameters because we don't want rf_obj to change
# when we generate a new GPInference object
@dataclass(kwargs_only=True, frozen=True)
class ThompsonState:
    dataset: ThompsonDataset
    rf_obj: RandomFeatures
    w_true: torch.Tensor  # Underlying weights used to generate the objective function
    fn_max: float
    fn_argmax: int


class BayesOpt:
    def __init__(
        self, bo_config: BayesOptConfig, device: torch.device, dtype: torch.dtype
    ):
        # Unpack inputs from bo_config
        self.min_val = bo_config.min_val
        self.max_val = bo_config.max_val
        self.dim = bo_config.dim
        self.kernel_config = bo_config.kernel_config
        self.noise_variance = bo_config.noise_variance
        self.sampling_method = bo_config.sampling_method
        self.num_init_samples = bo_config.num_init_samples
        self.acquisition_opt_config = bo_config.acquisition_opt_config

        self.device = device
        self.dtype = dtype

        # Get initialization points
        x_init = self._get_x_init()  # noqa: F841

    def _get_x_init(self):
        # Sample intializaiton points uniformly from the domain
        slope = self.max_val - self.min_val
        intercept = self.min_val
        x_init = torch.rand(
            self.num_init_samples, self.dim, device=self.device, dtype=self.dtype
        )
        return slope * x_init + intercept
