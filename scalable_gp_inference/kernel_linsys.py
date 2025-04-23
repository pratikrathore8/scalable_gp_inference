from rlaopt.models import LinSys
from rlaopt.kernels import KernelConfig
from rlaopt.solvers import _get_solver_name, _is_solver_config
from rlaopt.utils import Logger, _is_torch_tensor

import torch

from .sdd_config import SDDConfig
from .utils import _get_kernel_linop, get_solver


class KernelLinSys(LinSys):
    """Kernel ridge regression linear system."""

    def __init__(
        self,
        X: torch.Tensor,
        B: torch.Tensor,
        reg: float,
        kernel_type: str,
        kernel_config: KernelConfig,
        residual_tracking_idx: torch.Tensor | None = None,
        distributed: bool = False,
        devices: set[torch.device] | None = None,
    ):
        """Initialize KernelLinSys model.

        Args:
            X (torch.Tensor): Input data.
            B (torch.Tensor): Right-hand side.
            reg (float): Regularization parameter.
            kernel_type (str): Type of kernel.
            kernel_config (KernelConfig): Kernel configuration.
            residual_tracking_idx (torch.Tensor | None): Indices of columns of B to
            track for termination. If None, all columns are tracked.
            Defaults to None.
            distributed (bool): Whether to use distributed computation.
            Defaults to False.
            devices (set[torch.device] | None): Set of devices to use for distributed
            computation. Defaults to None.
        """
        # Set up superclass
        kernel_linop = _get_kernel_linop(
            X, X, kernel_type, kernel_config, distributed, devices
        )
        super().__init__(
            A=kernel_linop,
            B=B,
            reg=reg,
            A_row_oracle=kernel_linop.row_oracle,
            A_blk_oracle=kernel_linop.blk_oracle,
        )
        # Figure out which columns of B to track for termination
        self.residual_tracking_idx = residual_tracking_idx
        self.B_eval = (
            B if residual_tracking_idx is None else B[:, self.residual_tracking_idx]
        )

    def _compute_internal_metrics(self, W: torch.Tensor):
        W_in = (
            W[:, self.residual_tracking_idx]
            if self.residual_tracking_idx is not None
            else W
        )
        abs_res = torch.linalg.norm(
            self.B_eval - (self.A @ W_in + self.reg * W_in), dim=0, ord=2
        )
        rel_res = abs_res / torch.linalg.norm(self.B_eval, dim=0, ord=2)
        return {"abs_res": abs_res, "rel_res": rel_res}

    def _check_termination_criteria(
        self, internal_metrics: dict, atol: float, rtol: float
    ):
        abs_res = internal_metrics["abs_res"]
        comp_tol = torch.clamp(
            rtol * torch.linalg.norm(self.B_eval, dim=0, ord=2), min=atol
        )
        self._mask = abs_res > comp_tol
        return (~self._mask).all().item()

    def solve(
        self,
        solver_config,
        W_init,
        callback_fn=None,
        callback_args=[],
        callback_kwargs={},
        callback_freq=10,
        log_in_wandb=False,
        wandb_init_kwargs=None,
    ):
        if not isinstance(solver_config, SDDConfig):
            _is_solver_config(solver_config, "solver_config")
            solver_name = _get_solver_name(solver_config)
        else:
            solver_name = "sdd"
        _is_torch_tensor(W_init, "W_init")
        if log_in_wandb and wandb_init_kwargs is None:
            raise ValueError(
                "wandb_init_kwargs must be specified if log_in_wandb is True"
            )

        # Termination criteria
        atol, rtol = solver_config.atol, solver_config.rtol

        def termination_fn(internal_metrics):
            return self._check_termination_criteria(internal_metrics, atol, rtol)

        # Setup logging
        log_fn = self._get_log_fn(callback_fn, callback_args, callback_kwargs)

        wandb_kwargs = self._get_wandb_kwargs(
            log_in_wandb=log_in_wandb,
            wandb_init_kwargs=wandb_init_kwargs,
            solver_name=solver_name,
            solver_config=solver_config,
            callback_freq=callback_freq,
        )

        logger = Logger(
            log_freq=callback_freq,
            log_fn=log_fn,
            wandb_kwargs=wandb_kwargs,
        )

        # Get solver
        solver = get_solver(lin_sys=self, W_init=W_init, solver_config=solver_config)

        # Run solver
        solution, log = self._train(
            logger=logger,
            termination_fn=termination_fn,
            solver=solver,
            max_iters=solver_config.max_iters,
        )

        return solution, log
