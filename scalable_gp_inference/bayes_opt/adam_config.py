from dataclasses import asdict, dataclass


@dataclass(kw_only=True)
class ClippingConfig:
    min_val: float = 0.0
    max_val: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(kw_only=True)
class AdamConfig:
    step_size: float = 1e-3
    beta_1: float = 0.9
    beta_2: float = 0.999
    clip: bool = True
    clipping_config: ClippingConfig = ClippingConfig()

    def to_dict(self) -> dict:
        return {
            "step_size": self.step_size,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "clip": self.clip,
            "clipping_config": self.clipping_config.to_dict(),
        }