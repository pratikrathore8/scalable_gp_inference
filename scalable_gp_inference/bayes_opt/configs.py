from dataclasses import dataclass
from rlaopt.kernels import KernelConfig
from rlaopt.solvers import SolverConfig
from .adam_config import AdamConfig


@dataclass(kw_only=True, frozen=True)
class BayesOptConfig:
    min_val: float = 0.0
    max_val: float = 1.0
    dim: int = 8
    kernel_config: KernelConfig
    noise_variance: float = 1e-6
    sampling_method: str = "uniform"  # or "trunc_normal"
    # Number of samples for initialization
    num_init_samples: int = 50000
    # Optimization configuration for solving KRR linear systems to compute the posterior
    krr_solver_config: SolverConfig
    # Acquisition function optimization configuration
    acquisition_opt_config: AdamConfig = AdamConfig(step_size=1e-3)


@dataclass(kw_only=True, freeze=True)
class TSConfig:
    # Number of iterations to run Thompson sampling in Bayesian optimization
    num_iters: int = 30

    # Parameters for the posterior
    num_random_features: int = 5000

    # Exploration to exploitation proportion
    exp_proportion: float = 0.1

    # Exploration method
    exp_method: str = "nearby"  # or "uniform"

    # Number of exploration iterations (per full TS iteration)
    num_exp_iters: int = 30

    # Number of exploration points (per exploration iteration)
    num_exp_samples: int = 50000

    # Number of points to keep after exploring (per exploration iteration)
    num_top_exploration_points: int = 1

    # Number of points to keep per acquisition function
    # after all explorations (per full TS iteration)
    num_top_acquisition_points: int = 1

    # Number of acquisitions from the posterior (per full TS iteration)
    num_acquisitions: int = 1000
