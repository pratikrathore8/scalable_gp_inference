from typing import NamedTuple

import torch

from .ts_config import TSConfig
from .ts_utils import ThompsonDataset, _featurize, get_thompson_sampling_dataset


class ThompsonState(NamedTuple):
    ds: ThompsonDataset
    feature_params: torch.Tensor
    true_w: torch.Tensor
    max_fn_value: float
    argmax: float
    noise_scale: float


def init_state(config: TSConfig):
    (
        ds_init,
        params,
        w,
        max_fn_value,
        argmax,
        noise_scale,
    ) = get_thompson_sampling_dataset(config)

    return ThompsonState(
        ds=ds_init,
        feature_params=params,
        true_w=w,
        max_fn_value=max_fn_value,
        argmax=argmax,
        noise_scale=noise_scale,
    )


def update_state(state: ThompsonState, x_maxim: torch.Tensor):
    """Update the current state by
    - adding 'x_maxim' and corresponding objective function values to the state
    - adding 'x_maxim' and 'y_maxim' to data of the state
    - adding features for 'x_maxim' to L
    - replacing current 'argmax' and 'max_fn_value' if a new maximum has been found
    """
    y_maxim = _featurize(x_maxim, state.feature_params) @ state.true_w
    y_maxim = y_maxim + torch.randn(y_maxim.shape) * state.noise_scale

    # add besties to state dataset
    x = torch.concatenate([state.ds.x, x_maxim], axis=0)
    y = torch.concatenate([state.ds.y, y_maxim], axis=0)
    N, D = x.shape
    # construct updated state dataset
    ds = ThompsonDataset(x=x, y=y, N=N, D=D)

    # find maximum of current x_maxim, y_maxim
    idx = torch.argmax(y_maxim)
    argmax, max_fn_value = x_maxim[idx], y_maxim[idx]
    # update maximum in state if appropriate

    # max_fn_value, argmax = jax.lax.cond(
    #     max_fn_value <= state.max_fn_value,
    #     lambda: (state.max_fn_value, state.argmax),
    #     lambda: (max_fn_value, argmax),
    # )

    # construct and return updated state
    updated_state = ThompsonState(
        ds=ds,
        feature_params=state.feature_params,
        true_w=state.true_w,
        max_fn_value=max_fn_value,
        argmax=argmax,
        noise_scale=state.noise_scale,
    )
    return updated_state
