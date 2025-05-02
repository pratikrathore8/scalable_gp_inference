from dataclasses import dataclass
from rlaopt.kernels import KernelConfig
from .adam_config import AdamConfig


@dataclass(kw_only=True)
class TSConfig:
    # Dataset params
    ntr: int = 50000
    D: int = 8
    min_val: float = 0.0
    max_val: float = 1.0

    # Random features params
    num_features: int = 5000
    signal_scale = 1.0
    length_scale = 0.5
    noise_scale = 1e-3

    ts_iterations: int
    n_samples: int

    n_exp_pts: int = 30000
    n_greedy_pts: int = 1
    n_maxim: int = 1
    exploration_iters: int = 30

    # Acquisition function optimizer
    opt_config: AdamConfig = AdamConfig()

    # kernel params
    kernel_type: str
    kernel_config: KernelConfig
