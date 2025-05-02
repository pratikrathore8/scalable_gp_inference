from typing import Callable, NamedTuple

from adam_config import AdamConfig

import torch


class AdamState(NamedTuple):
    beta_1: float
    beta_2: float
    step_size: float
    m: float
    v: float
    iter_count: int = 0
    damping: float = 1e-8
    clip_max_val: float = None
    clip_min_val: float = None


def init_adam(
    params: torch.Tensor, config: AdamConfig
) -> tuple[
    Callable[[torch.Tensor, AdamState, torch.Tensor], tuple[torch.Tensor, AdamState]],
    AdamState,
]:

    state = _init_state(params, config)
    if config.clip is True:
        return _clipped_step, state
    else:
        return _step, state


def _init_state(params: torch.Tensor, config: AdamConfig) -> tuple[AdamState]:

    state = AdamState(
        beta_1=config.beta_1,
        beta_2=config.beta_2,
        step_size=config.step_size,
        m=torch.zeros_like(params, device=params.device),
        v=torch.zeros_like(params, device=params.device),
    )

    if config.clip is False:
        return state
    else:
        return state._replace(
            clip_max_val=config.clipping_config.max_val,
            clip_min_val=config.clipping_config.min_val,
        )


def _clipped_step(params: torch.Tensor, state: AdamState, grads: torch.Tensor):
    params, state = _step(params, state, grads)
    return params.clip(min=state.clip_min_val, max=state.clip_min_val), state


def _step(params: torch.Tensor, state: AdamState, grads: torch.Tensor):
    updates, state = _update(grads, state)
    params = _apply_updates(updates, params, state)
    return params, state


def _apply_updates(updates: torch.Tensor, params: torch.Tensor, state: AdamState):
    return params - state.step_size * updates


def _update(grads: torch.Tensor, state: AdamState) -> tuple[torch.Tensor, AdamState]:

    m = state.beta_1 * state.m + (1 - state.beta_2) * grads
    v = state.beta_2 * state.v + (1 - state.beta_2) * grads**2

    m_hat = m / (1 - state.beta_1 ** (state.iter_count + 1))
    v_hat = v / (1 - state.beta_2 ** (state.iter_count + 1))

    updates = m_hat / (torch.sqrt(v_hat) + state.damping)

    return updates, state._replace(m=m, v=v, iter_count=state.iter_count + 1)
