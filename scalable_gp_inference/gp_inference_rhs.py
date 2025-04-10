from typing import Union

import torch
from gpytorch.kernels import Kernel

from .enums import _StackingMode


class _TensorKernelStack:
    def __init__(
        self,
        y: torch.Tensor,
        X_kernel: Union[torch.Tensor, Kernel],
        stacking_mode: _StackingMode,
    ):
        if y.ndim != 2:
            raise ValueError(f"y must be 2D, but got {y.ndim}D")
        if X_kernel.ndim != 2:
            raise ValueError(f"X_kernel must be 2D, but got {X_kernel.ndim}D")
        if y.shape[0] != X_kernel.shape[0]:
            raise ValueError(
                f"y.shape[0] ({y.shape[0]}) must match "
                f"X_kernel.shape[0] ({X_kernel.shape[0]})"
            )
        if y.device != X_kernel.device:
            raise ValueError(
                f"y.device ({y.device}) must match X_kernel.device ({X_kernel.device})"
            )
        if y.dtype != X_kernel.dtype:
            raise ValueError(
                f"y.dtype ({y.dtype}) must match X_kernel.dtype ({X_kernel.dtype})"
            )

        self.y = y
        self.X_kernel = X_kernel
        self.stacking_mode = stacking_mode

    @property
    def device(self) -> torch.device:
        return self.y.device

    @property
    def shape(self) -> torch.Size:
        return torch.Size((self.y.shape[0], self.y.shape[1] + self.X_kernel.shape[1]))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __matmul__(self, w: torch.Tensor) -> torch.Tensor:
        unsqueeze = True if w.ndim == 1 else False
        if unsqueeze:
            w = w.unsqueeze(-1)

        # Compute the matmul using a block matrix multiplication
        y_cols = self.y.shape[1]
        result = self.y @ w[:y_cols] + self.X_kernel @ w[y_cols:]
        return result.squeeze(-1) if unsqueeze else result

    # def __rmatmul__(self, w: torch.Tensor) -> torch.Tensor:
    #     unsqueeze = True if w.ndim == 1 else False
    #     if unsqueeze:
    #         w = w.unsqueeze(-1)

    #     result = torch.cat((w @ self.y, w @ self.X_kernel), dim=1)


class GPInferenceRHS(_TensorKernelStack):
    def __init__(self, y: torch.Tensor, X_kernel: Union[torch.Tensor, Kernel]):
        super().__init__(y, X_kernel, _StackingMode.HORIZONTAL)
