from dataclasses import dataclass
from typing import Callable

import torch
from torch.func import grad, vmap
from rlaopt.solvers import SolverConfig

from ..kernel_linsys import KernelLinSys
from ..utils import _get_kernel_linop, _safe_unsqueeze
from ..random_features import RFConfig, RandomFeatures, get_prior_samples
from .configs import BayesOptConfig, TSConfig
from .adam_func import init_adam


@dataclass(kw_only=True, frozen=False)
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
) -> torch.Tensor:
    y = rf_obj.get_random_features(X) @ w
    y += torch.randn_like(y) * (noise_variance**0.5)
    return y


def _max_y(y: torch.Tensor) -> tuple[float, int]:
    y_max, y_argmax = torch.max(y, dim=0)
    return y_max.item(), y_argmax.item()


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
        X_init = self._sample_uniformly_from_domain(self.num_init_samples)

        # Setup random features
        rf_config = RFConfig(
            num_features=self.num_random_features, regenerate=False, in_place_ops=False
        )
        rf_obj = RandomFeatures(self.kernel_config, self.kernel_type, rf_config)

        # Form function that we will optimize via approximate posterior sampling
        w_true = torch.randn(
            self.num_random_features, device=self.device, dtype=self.dtype
        )
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

    def _sample_uniformly_from_domain(self, num_samples: int) -> torch.Tensor:
        slope = self.max_val - self.min_val
        intercept = self.min_val
        samples = torch.rand(
            num_samples, self.dim, device=self.device, dtype=self.dtype
        )
        return slope * samples + intercept

    def _get_exploration_points(
        self, num_samples: int, method: str, exploration_proportion: float
    ) -> torch.Tensor:
        if method == "uniform":
            return self._sample_uniformly_from_domain(num_samples)
        elif method == "nearby":
            # In this context, num_explore is really
            # how many samples we want to be uniformly random.
            # num_exploit is how many samples we want to draw
            # based on the evaluated objectives
            num_explore = int(num_samples * exploration_proportion)
            num_exploit = num_samples - num_explore

            X_list = []

            if num_explore > 0:
                X_explore = self._sample_uniformly_from_domain(num_explore)
                X_list.append(X_explore)

            if num_exploit > 0:
                localized_noise = torch.randn(
                    num_exploit, self.dim, dtype=self.dtype, device=self.device
                )
                localized_noise *= self.kernel_config.lengthscale / 2

                # NOTE(pratik): In the implementation by Lin et al. (2023),
                # the - on the minimum is a +, but this could
                # lead to negative values for the scores...
                scores = self.ts_state.y - self.ts_state.y.min() + 1e-6
                sampling_idxs = torch.multinomial(
                    scores, num_samples=num_exploit, replacement=True
                )
                X_exploit = self.ts_state.X[sampling_idxs] + localized_noise
                X_list.append(X_exploit)

            # Combine explore + exploit points and ensure they lie within the domain
            X_combined = torch.cat(X_list, dim=0)
            X_combined = torch.clamp(X_combined, min=self.min_val, max=self.max_val)
            return X_combined
        else:
            raise ValueError("method must be one of 'uniform' or 'nearby'.")

    def _get_acquisition_fn(
        self,
        alpha_obj: torch.Tensor,
        alpha_samples: torch.Tensor,
        w_samples: torch.Tensor,
    ) -> tuple[Callable, Callable, Callable]:
        if alpha_samples.dim() > 2:
            alpha_samples = alpha_samples.squeeze()
        if w_samples.dim() > 2:
            w_samples = w_samples.squeeze()

        # Necessary to make vmap work
        alpha_samples = alpha_samples.T

        def _fn(x, alpha_sample, w_sample):
            # x: (D,)
            # alpha_sample: (n_train,)
            # w_sample: (n_features,)
            # return: ()
            L = self.ts_state.rf_obj.get_random_features(x)
            K = _get_kernel_linop(
                x.unsqueeze(0),
                self.ts_state.X,
                self.kernel_type,
                self.kernel_config,
                distributed=False,
            )
            return (L @ w_sample + K @ (alpha_obj - alpha_sample)).squeeze()

        def acquisition_fn_sharex(x):
            """
            in_shape: (n_inputs, D)
            out_shape: (n_samples, n_inputs)
            """
            return vmap(vmap(_fn, in_dims=(0, None, None)), in_dims=(None, 0, 0))(
                x, alpha_samples, w_samples
            )

        def acquisition_fn(x):
            """
            in_shape: (n_samples, n_inputs, D)
            out_shape: (n_samples, n_inputs)
            """
            return vmap(vmap(_fn, in_dims=(0, None, None)), in_dims=(0, 0, 0))(
                x, alpha_samples, w_samples
            )

        def acquisition_grad(x):
            """
            in_shape: (n_samples, n_inputs, D)
            out_shape: (n_samples, n_inputs, D)
            """
            grad_fn = grad(_fn)
            return torch.transpose(
                vmap(vmap(grad_fn, in_dims=(0, 0, 0)), in_dims=(1, None, None))(
                    x, alpha_samples, w_samples
                ),
                0,
                1,
            )

        return acquisition_fn_sharex, acquisition_fn, acquisition_grad

    def _get_top_acquisition_points(
        self,
        top_exploration_points: torch.Tensor,
        acquisition_fn: Callable,
        acquisition_grad: Callable,
        num_top_acquisition_points: int,
    ) -> torch.Tensor:
        step, state = init_adam(top_exploration_points, self.acquisition_opt_config)
        for _ in range(self.num_acquisition_opt_iters):
            grads = acquisition_grad(top_exploration_points)
            top_exploration_points, state = step(top_exploration_points, state, -grads)

        y_top_exploration = acquisition_fn(top_exploration_points)

        _, top_acquisition_points_idx = torch.topk(
            y_top_exploration, k=num_top_acquisition_points, dim=1
        )

        # Use advanced indexing to select the top points correctly
        batch_indices = torch.arange(top_exploration_points.shape[0]).unsqueeze(1)
        acquisition_points = top_exploration_points[
            batch_indices, top_acquisition_points_idx
        ]

        # Reshape acquisition points so they are two-dimensional
        acquisition_points = acquisition_points.reshape(
            -1, acquisition_points.shape[-1]
        )

        return acquisition_points

    def _gp_sample_argmax(
        self,
        alpha_obj: torch.Tensor,
        alpha_samples: torch.Tensor,
        w_samples: torch.Tensor,
        ts_config: TSConfig,
    ) -> torch.Tensor:
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
                ts_config.num_exp_samples,
                method=ts_config.exp_method,
                exploration_proportion=ts_config.exp_proportion,
            )

            # Second, evaluate the acquisition functions at these candidate points
            y_exploration = acquisition_fn_sharex(exploration_points)

            # Now, find the top candidate points based on
            # the evaluated acquisition functions
            _, top_exploration_points_idx = torch.topk(
                y_exploration, k=ts_config.num_top_exp_points, dim=1
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

    def _update_state(self, top_acquisition_points: torch.Tensor):
        # Evaluate objective at top acquisition points
        y_top = _eval_y(
            top_acquisition_points,
            self.ts_state.rf_obj,
            self.ts_state.w_true,
            self.noise_variance,
        )

        # Find maximum of newly evaluated points
        y_top_max, y_top_argmax = _max_y(y_top)

        # If the maximum over newly evaluated points is larger than
        # everything we have seen then update fn_max and fn_argmax in the state
        # We do this before updating X and y in the state to ensure fn_argmax
        # is updated correctly
        if y_top_max > self.ts_state.fn_max:
            self.ts_state.fn_max = y_top_max
            self.ts_state.fn_argmax = len(self.ts_state) + y_top_argmax

        # Update state based on newly evaluated objective values
        self.ts_state.X = torch.cat((self.ts_state.X, top_acquisition_points), dim=0)
        self.ts_state.y = torch.cat((self.ts_state.y, y_top), dim=0)

    def step(self, ts_config: TSConfig, krr_solver_config: SolverConfig | None = None):
        if ts_config.acquisition_method == "random_search":
            # If we are acquiring radomly, get acquisition points
            # for all acquisition functions in one go
            acquisition_points = self._get_exploration_points(
                ts_config.num_acquisition_fns * ts_config.num_top_acquisition_points,
                method="uniform",
                exploration_proportion=None,
            )
        elif ts_config.acquisition_method == "gp":
            if krr_solver_config is None:
                raise ValueError(
                    "krr_solver_config cannot be None when acquisition method is 'gp'."
                )

            # Get prior samples, which we will use to sample from the posterior
            prior_samples, w_samples = get_prior_samples(
                X=self.ts_state.X,
                rf_obj=self.ts_state.rf_obj,
                noise_variance=self.noise_variance,
                num_samples=ts_config.num_acquisition_fns,
                return_feature_weights=True,
            )

            # Form KRR linear system which we use to solve
            # for alpha_obj and alpha_samples
            krr_linsys = KernelLinSys(
                X=self.ts_state.X,
                B=torch.cat(
                    (_safe_unsqueeze(self.ts_state.y), _safe_unsqueeze(prior_samples)),
                    dim=1,
                ),
                reg=self.noise_variance,
                kernel_type=self.kernel_type,
                kernel_config=self.kernel_config,
                use_full_kernel=True,
            )
            alpha_all, _ = krr_linsys.solve(
                solver_config=krr_solver_config, W_init=torch.zeros_like(krr_linsys.B)
            )

            # final_log_idx = list(log.keys())[-1]
            # print(log[final_log_idx]["metrics"]["internal_metrics"]["rel_res"])

            alpha_obj = alpha_all[:, 0]
            alpha_samples = alpha_all[:, 1:]

            # Get the acquisition points using _gp_sample_argmax and update the state
            acquisition_points = self._gp_sample_argmax(
                alpha_obj, alpha_samples, w_samples, ts_config
            )
        else:
            raise ValueError(
                "acquisition_method must be one of 'random_search' or 'gp'."
            )

        self._update_state(acquisition_points)
