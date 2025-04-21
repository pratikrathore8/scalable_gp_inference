from typing import Optional, Set, Union


from rlaopt.solvers import SolverConfig
import torch

from .kernel_linsys import KernelLinSys
from .random_features import get_random_features
from .utils import _get_kernel_linop, _safe_unsqueeze


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
        num_posterior_samples: Optional[int] = 0,
        num_random_features: Optional[int] = 0,
    ):
        # NOTE(pratik): this class assumes a zero-mean GP prior
        self.Xtr = Xtr
        self.ytr = ytr
        self.Xtst = Xtst
        self.ytst = ytst
        self.noise_variance = noise_variance
        self.kernel_type = kernel_type
        self.kernel_lengthscale = kernel_lengthscale
        self.distributed = distributed
        self.devices = devices
        self.num_posterior_samples = num_posterior_samples
        self.num_random_features = num_random_features
        (
            self.Xtr_prior_samples,
            self.Xtst_prior_samples,
        ) = self._get_approx_prior_samples()

    def _get_approx_prior_samples(self):
        if self.num_posterior_samples > 0 and self.num_random_features > 0:
            X = torch.cat((self.Xtr, self.Xtst), dim=0)
            Xtr_prior_samples = torch.zeros(
                self.Xtr.shape[0], self.num_posterior_samples
            )
            Xtst_prior_samples = torch.zeros(
                self.Xtst.shape[0], self.num_posterior_samples
            )

            # TODO(pratik): eventually vectorize over the posterior samples
            for i in range(self.num_posterior_samples):
                X_featurized = get_random_features(
                    X,
                    num_features=self.num_random_features,
                    lengthscale=self.kernel_lengthscale,
                    kernel_type=self.kernel_type,
                )
                w = torch.randn(X_featurized.shape[1], device=X.device, dtype=X.dtype)
                prior_samples = X_featurized @ w
                Xtr_prior_samples[:, i] = prior_samples[: self.Xtr.shape[0]] + (
                    self.noise_variance**0.5
                ) * torch.randn(
                    self.Xtr.shape[0], device=self.Xtr.device, dtype=self.Xtr.dtype
                )
                Xtst_prior_samples[:, i] = prior_samples[self.Xtr.shape[0] :]
            return Xtr_prior_samples, Xtst_prior_samples
        return (None,) * 2

    def _get_linsys(self):
        if self.Xtr_prior_samples > 0:
            B = torch.cat(
                (_safe_unsqueeze(self.ytr), _safe_unsqueeze(self.Xtr_prior_samples)),
                dim=1,
            )
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

    def _get_tst_kernel_linop(self):
        return _get_kernel_linop(
            self.Xtst,
            self.Xtr,
            kernel_type=self.kernel_type,
            kernel_lengthscale=self.kernel_lengthscale,
            distributed=self.distributed,
            devices=self.devices,
        )

    def _compute_nll(means: torch.Tensor, variances: torch.Tensor, locs: torch.Tensor):
        n = means.shape[0]
        log_variances = torch.log(variances)
        nll = 0.5 * (
            (locs - means) ** 2 / variances
            + torch.sum(log_variances)
            + n * torch.log(torch.tensor(2 * torch.pi))
        )
        return nll

    def _callback_fn(self, W: torch.Tensor, linsys: KernelLinSys, tst_kernel_linop):
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
            "test_mean": test_mean.cpu().numpy(),
        }

        # Compute variances and negative log likelihood using posterior samples
        if self.Xtr_prior_samples is not None and self.Xtst_prior_samples is not None:
            W_diff = _safe_unsqueeze(W[:, 0]) - W[:, 1:]
            test_posterior_samples = self.Xtst_prior_samples + tst_kernel_linop @ W_diff
            test_posterior_samples_mean = test_posterior_samples.mean(
                dim=1
            )  # Useful for sanity checking with test_mean
            test_posterior_samples_var = test_posterior_samples.var(dim=1)
            test_posterior_samples_nll = self._compute_nll(
                means=test_posterior_samples_mean,
                variances=test_posterior_samples_var,
                locs=self.ytst,
            )
            test_posterior_samples_mean_nll = (
                test_posterior_samples_nll / self.Xtst.shape[0]
            )
            mean_value = test_posterior_samples_mean.cpu().numpy()
            var_value = test_posterior_samples_var.cpu().numpy()
            nll_value = test_posterior_samples_nll.cpu().item()
            mean_nll_value = test_posterior_samples_mean_nll.cpu().item()
            metrics_dict.update(
                {
                    "test_posterior_samples_mean": mean_value,
                    "test_posterior_samples_var": var_value,
                    "test_posterior_samples_nll": nll_value,
                    "test_posterior_samples_mean_nll": mean_nll_value,
                }
            )
        return metrics_dict

    def perform_inference(
        self,
        solver_config: SolverConfig,
        W_init: Optional[torch.Tensor] = None,
        eval_freq: Optional[int] = 10,
        log_in_wandb: Optional[bool] = False,
        wandb_init_kwargs: Optional[dict] = {},
    ):
        linsys = self._get_linsys()
        tst_kernel_linop = self._get_tst_kernel_linop()

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
