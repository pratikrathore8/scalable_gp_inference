from typing import Dict, Optional, Set, Union

from rlaopt.solvers import SolverConfig
import torch
from torch.distributions import MultivariateNormal

from .utils import _get_kernel_linop, _get_kernel_diag, _safe_unsqueeze
from .kernel_linsys import KernelLinSys


class GPInference:
    def __init__(
        self,
        Xtr: torch.Tensor,
        ytr: torch.Tensor,
        Xtst: torch.Tensor,
        ytst: torch.Tensor,
        noise_variance: float,
        kernel_type: str,
        kernel_lengthscale: Union[float, torch.Tensor],
        distributed: Optional[bool] = False,
        devices: Optional[Set[torch.device]] = None,
        nll_test_samples: Optional[int] = 0,
    ):
        self.Xtr = Xtr
        self.ytr = ytr
        self.Xtst = Xtst
        self.ytst = ytst
        self.noise_variance = noise_variance
        self.kernel_type = kernel_type
        self.kernel_lengthscale = kernel_lengthscale
        self.distributed = distributed
        self.devices = devices
        self.nll_test_samples = nll_test_samples
        self.sample_idx, self.B_nll_test = self._get_B_nll_test()

    def _get_kernel_linop_internal(self, X, Y):
        return _get_kernel_linop(
            X,
            Y,
            kernel_type=self.kernel_type,
            kernel_lengthscale=self.kernel_lengthscale,
            distributed=self.distributed,
            devices=self.devices,
        )

    def _get_B_nll_test(self):
        if self.nll_test_samples > 0:
            sample_idx = torch.randperm(self.Xtst.shape[0])[: self.nll_test_samples]
            nll_test_linop = self._get_kernel_linop_internal(
                self.Xtr, self.Xtst[sample_idx]
            )
            B_nll_test = nll_test_linop @ torch.eye(
                sample_idx.shape[0], device=self.Xtst.device
            )
            del nll_test_linop
            return sample_idx, B_nll_test
        return (None,) * 2

    def _get_linsys(self):
        if self.B_nll_test is not None:
            B = torch.cat([_safe_unsqueeze(self.ytr), self.B_nll_test], dim=1)
        else:
            B = _safe_unsqueeze(self.ytr)

        return KernelLinSys(
            X=self.Xtr,
            B=B,
            reg=self.noise_variance,
            kernel_type=self.kernel_type,
            kernel_lengthscale=self.kernel_lengthscale,
            distributed=self.distributed,
            devices=self.devices,
        )

    def _callback_fn(self, W: torch.Tensor, linsys: KernelLinSys, tst_kernel_linop):
        # The first column of W is used to calculate the posterior mean
        # The remaining columns of W are used for calculating the posterior variance
        # corresponding to the self.nll_test_samples test points
        train_rmse = torch.sqrt(
            1 / self.ytr.shape[0] * torch.sum((self.ytr - linsys.A @ W[:, 0]) ** 2)
        )
        test_mean = tst_kernel_linop @ W[:, 0]
        test_rmse = torch.sqrt(
            1 / self.ytst.shape[0] * torch.sum((self.ytst - test_mean) ** 2)
        )
        metrics_dict = {
            "train_rmse": train_rmse.cpu().item(),
            "test_rmse": test_rmse.cpu().item(),
        }

        # Compute posterior variance at selected test points
        if self.B_nll_test is not None:
            test_mean_sampled = test_mean[self.sample_idx]
            # TODO(pratik): Fix the calculation of variance is currently incorrect
            test_var_sampled = _get_kernel_diag(
                self.Xtst[self.sample_idx], self.kernel_type, self.kernel_lengthscale
            ) - torch.sum(self.B_nll_test.T * W[:, 1:].T, dim=1)
            ytst_sampled = self.ytst[self.sample_idx]
            # Compute negative log likelihood
            test_nll_sampled = -MultivariateNormal(
                loc=test_mean_sampled, covariance_matrix=torch.diag(test_var_sampled)
            ).log_prob(ytst_sampled)
            test_mean_nll_sampled = test_nll_sampled / self.nll_test_samples
            metrics_dict.update(
                {
                    "test_mean_sampled": test_mean_sampled.cpu().numpy(),
                    "test_var_sampled": test_var_sampled.cpu().numpy(),
                    "test_nll_sampled": test_nll_sampled.cpu().item(),
                    "test_mean_nll_sampled": test_mean_nll_sampled.cpu().item(),
                }
            )
        return metrics_dict

    def perform_inference(
        self,
        solver_config: SolverConfig,
        W_init: Optional[torch.Tensor] = None,
        eval_freq: Optional[int] = 10,
        log_in_wandb: Optional[bool] = False,
        wandb_init_kwargs: Optional[Dict] = {},
    ):
        linsys = self._get_linsys()
        tst_kernel_linop = self._get_kernel_linop_internal(self.Xtst, self.Xtr)

        if W_init is None:
            W_init = torch.zeros_like(linsys.B)

        solution, log = linsys.solve(
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
