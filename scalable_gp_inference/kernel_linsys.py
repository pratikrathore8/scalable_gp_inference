from typing import Any, Optional, Set, Union

from rlaopt.models import LinSys
from rlaopt.kernels import (
    RBFLinOp,
    DistributedRBFLinOp,
    Matern12LinOp,
    DistributedMatern12LinOp,
    Matern32LinOp,
    DistributedMatern32LinOp,
    Matern52LinOp,
    DistributedMatern52LinOp,
)
import torch

from .gp_inference_rhs import GPInferenceRHS


class KernelLinSys(LinSys):
    """Kernel ridge regression linear system."""

    def __init__(
        self,
        X: torch.Tensor,
        B: Union[torch.Tensor, "GPInferenceRHS"],
        reg: float,
        kernel_type: str,
        kernel_lengthscale: float,
        residual_tracking_idx: Optional[torch.Tensor] = None,
        distributed: Optional[bool] = False,
        devices: Optional[Set[torch.device]] = None,
    ):
        """Initialize KernelLinSys model.

        Args:
            X (torch.Tensor): Input data.
            B (Union[torch.Tensor, "GPInferenceRHS"]): Right-hand side.
            reg (float): Regularization parameter.
            kernel_type (str): Type of kernel.
            kernel_lengthscale (float): Lengthscale for the kernel.
            residual_tracking_idx (Optional[torch.Tensor]): Residual tracking index.
            Defaults to None.
            distributed (bool): Whether to use distributed computation.
            Defaults to False.
            devices (Optional[Set[torch.device]]): Set of devices for
            distributed computation. Defaults to None.
        """
        # Set up superclass
        kernel_linop = self._get_kernel_linop(
            X, kernel_type, kernel_lengthscale, distributed, devices
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
        self.B_eval = (
            self.B_eval.to_dense() if isinstance(B, GPInferenceRHS) else self.B_eval
        )

    def _get_kernel_linop(
        self,
        X: torch.Tensor,
        kernel_type: str,
        kernel_lengthscale: float,
        distributed: bool,
        devices: Optional[Set[torch.device]],
    ):
        """Get the kernel linear operator class based on the kernel type."""
        if kernel_type == "rbf":
            linop_class = DistributedRBFLinOp if distributed else RBFLinOp
        elif kernel_type == "matern12":
            linop_class = DistributedMatern12LinOp if distributed else Matern12LinOp
        elif kernel_type == "matern32":
            linop_class = DistributedMatern32LinOp if distributed else Matern32LinOp
        elif kernel_type == "matern52":
            linop_class = DistributedMatern52LinOp if distributed else Matern52LinOp
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        if devices is None:
            devices = set([X.device])

        linop_kwargs = {"A": X, "kernel_params": {"sigma": kernel_lengthscale}}
        if distributed:
            linop_kwargs.update({"devices": devices})
        return linop_class(**linop_kwargs)

    def _check_inputs(
        self,
        A: Any,
        B: Any,
        reg: Any,
        A_row_oracle: Optional[Any],
        A_blk_oracle: Optional[Any],
    ):
        # Override the superclass method to avoid checking inputs
        # The reason we do this is to avoid errors when B is a GPInferenceRHS instance
        pass

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
