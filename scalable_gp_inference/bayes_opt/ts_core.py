from dataclasses import dataclass
from typing import Callable

import torch
from rlaopt.solvers import SolverConfig

from ..kernel_linsys import KernelLinSys
from ..random_features import RFConfig, RandomFeatures
from .configs import BayesOptConfig, TSConfig


@dataclass(kwargs_only=True, frozen=False)
class ThompsonState:
    """
    Represents the state for Thompson sampling in Bayesian optimization.

    Attributes:
        x (torch.Tensor): Queried points in the domain.
        y (torch.Tensor): Function values evaluated at the points in x.
        rf_obj (RandomFeatures): Object that is used to generate random features.
        w_true (torch.Tensor): Weights defining the function to be optimized.
        fn_max (float): Maximum function value found so far.
        fn_argmax (int): Index in x corresponding to the maximum function value.
    """

    X: torch.Tensor
    y: torch.Tensor
    rf_obj: RandomFeatures
    w_true: torch.Tensor  # Underlying weights used to generate the objective function
    fn_max: float
    fn_argmax: int

    def __len__(self):
        """
        Returns the number of observations and
        verifies that X and y have the same length.

        Returns:
            int: The number of observations (y.shape[0]).

        Raises:
            ValueError: If X.shape[0] is not equal to y.shape[0].
        """
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"Mismatch in number of observations: X has {self.X.shape[0]} rows, "
                f"y has {self.y.shape[0]} elements"
            )
        return self.y.shape[0]


def _eval_y(
    X: torch.Tensor, rf_obj: RandomFeatures, w: torch.Tensor, noise_variance: float
):
    y = rf_obj.get_random_features(X) @ w
    y += torch.randn_like(y) * (noise_variance**0.5)
    return y


def _max_y(y: torch.Tensor):
    y_max, y_argmax = torch.max(y)
    y_max = y_max.cpu().item()
    y_argmax = y_argmax[0].cpu().item()
    return y_max, y_argmax


class BayesOpt:
    def __init__(
        self, bo_config: BayesOptConfig, device: torch.device, dtype: torch.dtype
    ):
        # Unpack inputs from bo_config
        self.min_val = bo_config.min_val
        self.max_val = bo_config.max_val
        self.dim = bo_config.dim
        self.kernel_type = bo_config.kernel_type
        self.kernel_config = bo_config.kernel_config
        self.noise_variance = bo_config.noise_variance
        self.num_random_features = bo_config.num_random_features
        self.num_init_samples = bo_config.num_init_samples
        self.acquisition_opt_config = bo_config.acquisition_opt_config
        self.num_acquisition_opt_iters = bo_config.num_acquisition_opt_iters

        self.device = device
        self.dtype = dtype

        # Get initialization points
        X_init = self._get_x_init()

        # Setup random features
        rf_config = RFConfig(num_features=self.num_random_features, regenerate=False)
        rf_obj = RandomFeatures(self.kernel_config, self.kernel_type, rf_config)

        # Form function that we will optimize via approximate posterior sampling
        w_true = torch.randn(self.num_random_features)
        y_init = _eval_y(X_init, rf_obj, w_true, self.noise_variance)

        # Find max value and index corresponding to max value
        fn_max, fn_argmax = _max_y(y_init)

        # Initialize Thompson state
        self.ts_state = ThompsonState(
            X=X_init,
            y=y_init,
            rf_obj=rf_obj,
            w_true=w_true,
            fn_max=fn_max,
            fn_argmax=fn_argmax,
        )

    def _get_x_init(self) -> torch.Tensor:
        # Sample intializaiton points uniformly from the domain
        slope = self.max_val - self.min_val
        intercept = self.min_val
        x_init = torch.rand(
            self.num_init_samples, self.dim, device=self.device, dtype=self.dtype
        )
        return slope * x_init + intercept

    def _update_state(self, top_acquisition_points: torch.Tensor):
        # Evaluate objective at top acquisition points
        y_top = _eval_y(
            top_acquisition_points,
            self.ts_state.rf_obj,
            self.ts_state.w_true,
            self.noise_variance,
        )

        # Update state based on newly evaluated objective values
        self.ts_state.X = torch.cat((self.ts_state.X, top_acquisition_points), dim=0)
        self.ts_state.y = torch.cat((self.ts_state.y, y_top), dim=0)

        # Find maximum of newly evaluated points
        y_top_max, y_top_argmax = _max_y(y_top)

        # If the maxmimum over newly evaluated points is larger than
        # everything we have seen then update fn_max and fn_argmax in the state
        if y_top_max > self.ts_state.fn_max:
            self.ts_state.fn_max = y_top_max
            self.ts_state.fn_argmax = len(self.ts_state) + y_top_argmax

    def _get_exploration_points(self, num_samples: int, method: str) -> torch.Tensor:
        pass

    def _get_acquisition_fn(
        self,
        alpha_obj: torch.Tensor,
        alpha_samples: torch.Tensor,
        w_samples: torch.Tensor,
    ) -> tuple[Callable, Callable, Callable]:
        pass

    def _get_top_acquisition_points(
        self,
        top_exploration_points: torch.Tensor,
        acquisition_fn: Callable,
        acquisition_grad: Callable,
        num_top_acquisition_points: int,
    ) -> torch.Tensor:
        pass

    def _gp_sample_argmax(
        self,
        alpha_obj: torch.Tensor,
        alpha_samples: torch.Tensor,
        w_samples: torch.Tensor,
        ts_config: TSConfig,
    ):
        # We assume that the acquisition functions are already parallelized
        # over the number of acquisitions
        (
            acquisition_fn_sharex,
            acquisition_fn,
            acquisition_grad,
        ) = self._get_acquisition_fn(alpha_obj, alpha_samples, w_samples)

        # Getting top candidates for initializing optimizer for
        # maximizing acquisition functions
        # Initialize top_exploration_points as None
        top_exploration_points = None

        for _ in range(ts_config.num_exp_iters):
            # First, sample a bunch of candidate points
            exploration_points = self._get_exploration_points(
                ts_config.num_exp_samples, method=ts_config.exp_method
            )

            # Second, evaluate the acquisition functions at these candidate points
            y_exploration = acquisition_fn_sharex(exploration_points)

            # Now, find the top candidate points based on
            # the evaluated acquisition functions
            _, top_exploration_points_idx = torch.topk(
                y_exploration, k=ts_config.num_top_exp_points, dim=0
            )

            current_top_points = exploration_points[top_exploration_points_idx]

            if top_exploration_points is None:
                top_exploration_points = current_top_points
            else:
                # Concatenate to existing top points along dimension 1
                top_exploration_points = torch.cat(
                    [top_exploration_points, current_top_points], dim=1
                )

        # After the loop, top_exploration_points will have shape:
        # (num_samples, num_top_exp_points * num_exp_iters, dimension)

        # Use the top exploration points to get the top acquisition points
        return self._get_top_acquisition_points(
            top_exploration_points,
            acquisition_fn,
            acquisition_grad,
            ts_config.num_top_acquisition_points,
        )

    def step(self, ts_config: TSConfig, krr_config: SolverConfig | None = None):
        if ts_config.acquisition_method == "random_search":
            # If we are acquiring radomly, get acquisition points
            # for all acquisition functions in one go
            acquisition_points = self._get_exploration_points(
                ts_config.num_acquisitions * ts_config.num_top_acquisition_points,
                method="uniform",
            )
            self._update_state(acquisition_points)
        # else:
