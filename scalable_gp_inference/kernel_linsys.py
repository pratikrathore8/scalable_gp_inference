from typing import Optional, Set, Union

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
    """Kernel linear system for Gaussian process inference."""

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
        self.residual_tracking_idx = residual_tracking_idx

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
