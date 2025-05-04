from dataclasses import dataclass
from rlaopt.kernels import KernelConfig
from .adam_config import AdamConfig


@dataclass(kw_only=True, frozen=True)
class BayesOptConfig:
    min_val: float = 0.0
    max_val: float = 1.0
    dim: int = 8
    kernel_type: str
    kernel_config: KernelConfig
    noise_variance: float = 1e-6
    # Number of random features for approximating the posterior
    num_random_features: int = 5000
    # Number of samples for initialization
    num_init_samples: int = 50000
    # Acquisition function optimization configuration
    acquisition_opt_config: AdamConfig = AdamConfig(step_size=1e-3)
    num_acquisition_opt_iters: int = 100

    def __post_init__(self):
        if self.min_val >= self.max_val:
            raise ValueError(
                "self.min_val must be smaller than self.max_val! "
                f"Received self.min_val = {self.min_val}, self.max_val = {self.max_val}"
            )


@dataclass(kw_only=True, freeze=True)
class TSConfig:
    # Number of iterations to run Thompson sampling in Bayesian optimization
    num_iters: int = 30

    # Exploration to exploitation proportion
    exp_proportion: float = 0.1

    # Exploration method
    exp_method: str = "nearby"  # or "uniform"

    # Number of exploration iterations (per full TS iteration)
    num_exp_iters: int = 30

    # Number of exploration points (per exploration iteration)
    num_exp_samples: int = 50000

    # Number of points to keep after exploring (per exploration iteration)
    num_top_exp_points: int = 1

    # Number of points to keep per acquisition function
    # after all explorations (per full TS iteration)
    num_top_acquisition_points: int = 1

    # Number of acquisitions from the posterior (per full TS iteration)
    num_acquisitions: int = 1000

    # Method for acquisition (either random search or GP)
    acquisition_method: str = "gp"
