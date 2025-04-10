from typing import Union

import torch
from gpytorch.kernels import Kernel


class _TensorKernelStack:
    def __init__(
        self,
        y: torch.Tensor,
        X_kernel: Union[torch.Tensor, Kernel],
        stack_horizontal: bool,
    ):
        self._check_inputs(y, X_kernel, stack_horizontal)
        self.y = y
        self.X_kernel = X_kernel
        self.stack_horizontal = stack_horizontal

    def _check_inputs(
        self,
        y: torch.Tensor,
        X_kernel: Union[torch.Tensor, Kernel],
        stack_horizontal: bool,
    ):
        if y.ndim != 2:
            raise ValueError(f"y must be 2D, but got {y.ndim}D")
        if X_kernel.ndim != 2:
            raise ValueError(f"X_kernel must be 2D, but got {X_kernel.ndim}D")
        if y.device != X_kernel.device:
            raise ValueError(
                f"y.device ({y.device}) must match X_kernel.device ({X_kernel.device})"
            )
        if y.dtype != X_kernel.dtype:
            raise ValueError(
                f"y.dtype ({y.dtype}) must match X_kernel.dtype ({X_kernel.dtype})"
            )
        if stack_horizontal:
            if y.shape[0] != X_kernel.shape[0]:
                raise ValueError(
                    f"y.shape[0] ({y.shape[0]}) must match "
                    f"X_kernel.shape[0] ({X_kernel.shape[0]})"
                )
        else:
            if y.shape[1] != X_kernel.shape[1]:
                raise ValueError(
                    f"y.shape[1] ({y.shape[1]}) must match "
                    f"X_kernel.shape[1] ({X_kernel.shape[1]})"
                )

    @property
    def device(self) -> torch.device:
        return self.y.device

    def _shape_horizontal(self) -> torch.Size:
        return torch.Size((self.y.shape[0], self.y.shape[1] + self.X_kernel.shape[1]))

    def _shape_vertical(self) -> torch.Size:
        return torch.Size((self.y.shape[0] + self.X_kernel.shape[0], self.y.shape[1]))

    @property
    def shape(self) -> torch.Size:
        return (
            self._shape_horizontal()
            if self.stack_horizontal
            else self._shape_vertical()
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self.y.dtype

    @property
    def T(self) -> "_TensorKernelStack":
        return _TensorKernelStack(self.y.T, self.X_kernel.T, not self.stack_horizontal)

    def _matmul_horizontal(self, w: torch.Tensor) -> torch.Tensor:
        # Compute the matmul using a block matrix multiplication
        y_cols = self.y.shape[1]
        return self.y @ w[:, :y_cols] + self.X_kernel @ w[:, y_cols:]

    def _matmul_vertical(self, w: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.y @ w, self.X_kernel @ w), dim=0)

    def __matmul__(self, w: torch.Tensor) -> torch.Tensor:
        return (
            self._matmul_horizontal(w)
            if self.stack_horizontal
            else self._matmul_vertical(w)
        )

    def __rmatmul__(self, w: torch.Tensor) -> torch.Tensor:
        if w.ndim == 1:
            return self.T @ w
        elif w.ndim == 2:
            return (self.T @ w.T).T


class GPInferenceRHS(_TensorKernelStack):
    def __init__(self, y: torch.Tensor, X_kernel: Union[torch.Tensor, Kernel]):
        super().__init__(y, X_kernel, stack_horizontal=True)
