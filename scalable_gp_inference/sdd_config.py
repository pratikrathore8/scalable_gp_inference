from dataclasses import dataclass

from rlaopt.preconditioners import IdentityConfig
from rlaopt.utils import _is_pos_float, _is_pos_int


@dataclass(kw_only=True)
class SDDConfig:
    m: float
    step_size: float
    theta: float
    blk_size: int
    max_iters: int

    def __post_init__(self):
        _is_pos_float(self.m, "m")
        _is_pos_float(self.step_size, "step_size")
        _is_pos_float(self.theta, "theta")
        if self.m > 1.0:
            raise ValueError("Momentum parameter m must be less than 1!")
        if self.theta > 1.0:
            raise ValueError("Average parameter theta must be less than 1!")

        _is_pos_int(self.blk_size, "batch_size")
        _is_pos_int(self.max_iters, "max_iters")

        self.precond_config = IdentityConfig()
