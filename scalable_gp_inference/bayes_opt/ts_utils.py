from dataclasses import dataclass

import torch

# from .ts_config import TSConfig


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


# def get_thompson_sampling_dataset(config: TSConfig):
#     x_init = torch.random.uniform(
#         config.ntr, minval=config.min_val, maxval=config.max_val
#     )

#     w = torch.randn(shape=(config.num_features,))

#     # TODO get equivalent of:
#     # params = kernel.feature_params_fn(feature_key, n_features, x_init.shape[-1])
#     y_init = _featurize(x_init, config.params) @ w
#     y_init = y_init + torch.randn(y_init.shape) * config.noise_scale

#     ds_init = ThompsonDataset(x_init, y_init, config.ntr, config.D)
#     idx = torch.argmax(y_init)
#     argmax, max_fn_value = x_init[idx], y_init[idx]

#     return ds_init, config.params, w, max_fn_value, argmax, config.noise_scale


# # TODO find where the appropriate params are in our implementation
# def _featurize(x: torch.Tensor, params: torch.Tensor):
#     return (
#         params.signal_scale
#         * torch.sqrt(2.0 / params.M)
#         * torch.cos((x / params.length_scale) @ params.omega + params.phi)
#     )
