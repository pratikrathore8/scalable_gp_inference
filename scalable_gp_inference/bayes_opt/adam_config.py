from dataclasses import dataclass


@dataclass(kw_only=True)
class ClippingConfig:
    min_val: float = 0.0
    max_val: float = 1.0


@dataclass(kw_only=True)
class AdamConfig:
    step_size: float = 1e-3
    beta_1: float = 0.9
    beta_2: float = 0.999
    clip: bool = True
    clipping_config: ClippingConfig = ClippingConfig()
