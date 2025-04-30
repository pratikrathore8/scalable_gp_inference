from rlaopt.kernels import (
    RBFLinOp,
    DistributedRBFLinOp,
    Matern12LinOp,
    DistributedMatern12LinOp,
    Matern32LinOp,
    DistributedMatern32LinOp,
    Matern52LinOp,
    DistributedMatern52LinOp,
    KernelConfig,
)

from rlaopt.solvers import _get_solver

import torch

from .sdd_config import SDDConfig
from .sdd import SDD
from .tanimoto_kernel import TanimotoLinOp, DistributedTanimotoLinOp

_KERNEL_LINOP_CLASSES = {
    "rbf": RBFLinOp,
    "distributed_rbf": DistributedRBFLinOp,
    "matern12": Matern12LinOp,
    "distributed_matern12": DistributedMatern12LinOp,
    "matern32": Matern32LinOp,
    "distributed_matern32": DistributedMatern32LinOp,
    "matern52": Matern52LinOp,
    "distributed_matern52": DistributedMatern52LinOp,
    "tanimoto": TanimotoLinOp,
    "distributed_tanimoto": DistributedTanimotoLinOp,
}


def get_solver(lin_sys, W_init, solver_config):
    if isinstance(solver_config, SDDConfig):
        return SDD(
            solver_config, W_init=W_init, system=lin_sys, device=solver_config.device
        )
    else:
        return _get_solver(lin_sys, W_init, solver_config)


def _get_kernel_linop(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel_type: str,
    kernel_config: KernelConfig,
    distributed: bool,
    devices: set[torch.device] | None = None,
):
    """Get the kernel linear operator class based on the kernel type."""
    if distributed:
        kernel_type = "distributed_" + kernel_type
    linop_class = _KERNEL_LINOP_CLASSES.get(kernel_type, None)
    if linop_class is None:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    linop_kwargs = {
        "A1": X,
        "A2": Y,
        "kernel_config": kernel_config,
    }
    if distributed:
        devices = set([X.device]) if devices is None else devices
        linop_kwargs.update({"devices": devices})
    return linop_class(**linop_kwargs)


def _safe_unsqueeze(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure the tensor is 2D by unsqueezing if necessary."""
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    return tensor


def _get_r2(X: torch.Tensor, y: torch.Tensor):
    # Fit a linear model
    X = _safe_unsqueeze(X)
    y = _safe_unsqueeze(y)
    X = torch.cat((torch.ones(X.shape[0], 1, device=X.device), X), dim=1)
    residuals = torch.linalg.lstsq(X, y).residuals

    mu_y = torch.mean(y, dim=0)
    ss_total = torch.sum((y - mu_y) ** 2, dim=0)
    ss_residuals = torch.sum(residuals, dim=0)
    r_squared = 1 - ss_residuals / ss_total
    return r_squared
