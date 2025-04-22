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

_KERNEL_LINOP_CLASSES = {
    "rbf": RBFLinOp,
    "distributed_rbf": DistributedRBFLinOp,
    "matern12": Matern12LinOp,
    "distributed_matern12": DistributedMatern12LinOp,
    "matern32": Matern32LinOp,
    "distributed_matern32": DistributedMatern32LinOp,
    "matern52": Matern52LinOp,
    "distributed_matern52": DistributedMatern52LinOp,
}


def get_solver(lin_sys, W_init, config):
    if isinstance(config, SDDConfig):
        return SDD(config, W_init=W_init, system=lin_sys, device=config.device)
    else:
        return _get_solver(lin_sys, W_init, config)


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
