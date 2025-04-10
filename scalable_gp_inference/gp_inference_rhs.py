from gpytorch.kernels import Kernel
from linear_operator.operators import DenseLinearOperator, CatLinearOperator
import torch


class GPInferenceRHS(CatLinearOperator):
    def __init__(
        self,
        targets: torch.Tensor,
        X: torch.Tensor,
        kernel_fn: Kernel,
        tst_idx: torch.Tensor,
    ):
        if targets.shape[0] != X.shape[0]:
            raise ValueError("The number of rows in targets and X must match.")

        targets_linop = DenseLinearOperator(targets)
        X_tst_kernel_linop = kernel_fn(X, X[tst_idx])
        super().__init__(targets_linop, X_tst_kernel_linop, dim=1)
