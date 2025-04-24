from dataclasses import dataclass

from rlaopt.preconditioners import IdentityConfig
from rlaopt.utils import _is_pos_float, _is_pos_int, _is_torch_device

import torch


@dataclass(kw_only=True)
class SDDConfig:
    momentum: float
    step_size: float
    theta: float
    blk_size: int
    max_iters: int
    device: torch.device
    atol: float = 1e-6
    rtol: float = 1e-6

    def __post_init__(self):
        _is_pos_float(self.momentum, "m")
        _is_pos_float(self.step_size, "step_size")
        _is_pos_float(self.theta, "theta")
        if self.momentum > 1.0:
            raise ValueError("momentum must be less than 1!")
        if self.theta > 1.0:
            raise ValueError("Average parameter theta must be less than 1!")

        _is_pos_int(self.blk_size, "batch_size")
        _is_pos_int(self.max_iters, "max_iters")
        _is_torch_device(self.device, "device")

        self.precond_config = IdentityConfig()
