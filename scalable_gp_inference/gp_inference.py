from typing import Optional, Set, Union

import torch
from rlaopt.solvers import SolverConfig

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
        self.ytr = ytr
        self.Xtst = Xtst
        self.ytst = ytst
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

    def _callback_fn(self, W: torch.Tensor, linsys: KernelLinSys):
        train_rmse = torch.sqrt(
            1 / linsys.A.shape[0] * torch.sum((linsys.B - linsys.A @ W) ** 2)
        )
        return {"train_rmse": train_rmse.cpu().item()}

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

        linsys = self._get_linsys()
        solution, log = linsys.solve(
            solver_config=solver_config,
            W_init=W_init,
            callback_fn=self._callback_fn,
            callback_args=[],
            callback_kwargs={},
            callback_freq=eval_freq,
            log_in_wandb=log_in_wandb,
            wandb_init_kwargs=wandb_init_kwargs,
        )

        return {"W_star": solution, "log": log}
