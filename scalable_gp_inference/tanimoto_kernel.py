from dataclasses import dataclass

import torch
from pykeops.torch import LazyTensor
from rlaopt.kernels import KernelConfig
from rlaopt.kernels.factory import _create_kernel_classes
from rlaopt.utils import _is_float


@dataclass(kw_only=True, frozen=False)
class TanimotoKernelConfig(KernelConfig):
    const_scaling: float
    lengthscale: float | torch.Tensor | None = None  # Not used in Tanimoto kernel

    def __post_init__(self):
        _is_float(self.const_scaling, "const_scaling")


def _kernel_computation_tanimoto(
    Ai_lazy: LazyTensor, Aj_lazy: LazyTensor, kernel_config: TanimotoKernelConfig
):
    # Compute (a-b) term
    diff = Ai_lazy - Aj_lazy

    # Get absolute difference
    abs_diff = diff.abs()

    # min(a,b) = (a+b-|a-b|)/2
    min_values = (Ai_lazy + Aj_lazy - abs_diff) / 2

    # max(a,b) = (a+b+|a-b|)/2
    max_values = (Ai_lazy + Aj_lazy + abs_diff) / 2

    # Sum along feature dimension
    min_sum = min_values.sum(dim=2)
    max_sum = max_values.sum(dim=2)

    # Compute Tanimoto coefficient
    tanimoto = min_sum / max_sum

    return kernel_config.const_scaling * tanimoto


TanimotoLinOp, DistributedTanimotoLinOp = _create_kernel_classes(
    kernel_name="Tanimoto",
    kernel_computation_fn=_kernel_computation_tanimoto,
)
