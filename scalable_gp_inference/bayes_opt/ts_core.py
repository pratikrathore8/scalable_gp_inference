from dataclasses import dataclass

import torch

from ..kernel_linsys import KernelLinSys
from ..random_features import RFConfig, RandomFeatures
from .configs import BayesOptConfig, TSConfig


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


@dataclass(kwargs_only=True, frozen=False)
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
        self.kernel_type = bo_config.kernel_type
        self.kernel_config = bo_config.kernel_config
        self.noise_variance = bo_config.noise_variance
        self.num_random_features = bo_config.num_random_features
        self.num_init_samples = bo_config.num_init_samples
        self.acquisition_opt_config = bo_config.acquisition_opt_config

        self.device = device
        self.dtype = dtype

        # Get initialization points
        x_init = self._get_x_init()
        rf_config = RFConfig(num_features=self.num_random_features, regenerate=False)
        rf_obj = RandomFeatures(self.kernel_config, self.kernel_type, rf_config)

    def _get_x_init(self):
        # Sample intializaiton points uniformly from the domain
        slope = self.max_val - self.min_val
        intercept = self.min_val
        x_init = torch.rand(
            self.num_init_samples, self.dim, device=self.device, dtype=self.dtype
        )
        return slope * x_init + intercept
