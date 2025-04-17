from typing import Optional, Set

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


def _get_kernel_linop(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel_type: str,
    kernel_lengthscale: float,
    distributed: bool,
    devices: Optional[Set[torch.device]] = None,
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
        "kernel_params": {"lengthscale": kernel_lengthscale},
    }
    if distributed:
        devices = set([X.device]) if devices is None else devices
        linop_kwargs.update({"devices": devices})
    return linop_class(**linop_kwargs)
