from dataclasses import dataclass

import torch

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
    noise_variance: float


class BayesOpt:
    def __init__(bo_config: BayesOptConfig):
        pass
