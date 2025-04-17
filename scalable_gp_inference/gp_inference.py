from typing import Optional, Set, Union

import torch
from rlaopt.solvers import SolverConfig

from .utils import _get_kernel_linop
from .kernel_linsys import KernelLinSys


class GPInference:
    def __init__(
        self,
        Xtr: torch.Tensor,
        ytr: torch.Tensor,
        Xtst: torch.Tensor,
        ytst: torch.Tensor,
        likelihood_variance: float,
        kernel_type: str,
        kernel_lengthscale: Union[float, torch.Tensor],
        distributed: Optional[bool] = False,
        devices: Optional[Set[torch.device]] = None,
    ):
        self.Xtr = Xtr
        self.ytr = ytr if ytr.ndim == 2 else ytr.unsqueeze(-1)
        self.Xtst = Xtst
        self.ytst = ytst if ytst.ndim == 2 else ytst.unsqueeze(-1)
        self.likelihood_variance = likelihood_variance
        self.kernel_type = kernel_type
        self.kernel_lengthscale = kernel_lengthscale
        self.distributed = distributed
        self.devices = devices

    def _get_linsys(self):
        return KernelLinSys(
            X=self.Xtr,
            B=self.ytr,
            reg=self.likelihood_variance,
            kernel_type=self.kernel_type,
            kernel_lengthscale=self.kernel_lengthscale,
            distributed=self.distributed,
            devices=self.devices,
        )

    def _get_tst_kernel_linop(self):
        return _get_kernel_linop(
            self.Xtst,
            self.Xtr,
            kernel_type=self.kernel_type,
            kernel_lengthscale=self.kernel_lengthscale,
            distributed=self.distributed,
            devices=self.devices,
        )

    def _callback_fn(self, W: torch.Tensor, linsys: KernelLinSys, tst_kernel_linop):
        train_rmse = torch.sqrt(
            1 / self.ytr.shape[0] * torch.sum((self.ytr - linsys.A @ W) ** 2)
        )
        test_rmse = torch.sqrt(
            1 / self.ytst.shape[0] * torch.sum((self.ytst - tst_kernel_linop @ W) ** 2)
        )
        return {
            "train_rmse": train_rmse.cpu().item(),
            "test_rmse": test_rmse.cpu().item(),
        }

    def perform_inference(
        self,
        solver_config: SolverConfig,
        W_init: Optional[torch.Tensor] = None,
        eval_freq: Optional[int] = 10,
        log_in_wandb: Optional[bool] = False,
        wandb_init_kwargs: Optional[dict] = None,
    ):
        if W_init is None:
            W_init = torch.zeros(self.X.shape[0], device=self.X.device)

        training_linsys = self._get_linsys()
        tst_kernel_linop = self._get_tst_kernel_linop()
        solution, log = training_linsys.solve(
            solver_config=solver_config,
            W_init=W_init,
            callback_fn=self._callback_fn,
            callback_args=[],
            callback_kwargs={"tst_kernel_linop": tst_kernel_linop},
            callback_freq=eval_freq,
            log_in_wandb=log_in_wandb,
            wandb_init_kwargs=wandb_init_kwargs,
        )

        return {"W_star": solution, "log": log}
