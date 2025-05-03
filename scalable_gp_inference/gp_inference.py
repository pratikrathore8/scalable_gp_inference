from rlaopt.solvers import SolverConfig
from rlaopt.kernels import KernelConfig
import torch

from .hparam_training import GPHparams
from .kernel_linsys import KernelLinSys
from .random_features import RFConfig, get_prior_samples
from .utils import _get_kernel_linop, _safe_unsqueeze, _get_r2


# def print_memory_usage(label=""):
#     allocated = torch.cuda.memory_allocated() / 1e9
#     reserved = torch.cuda.memory_reserved() / 1e9
#     print(f"{label} - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


class GPInference:
    def __init__(
        self,
        kernel_type: str,
        kernel_hparams: GPHparams,
        Xtr: torch.Tensor,
        ytr: torch.Tensor,
        Xtst: torch.Tensor | None = None,
        ytst: torch.Tensor | None = None,
        num_posterior_samples: int = 0,
        rf_config: RFConfig = RFConfig(num_features=0),
        distributed: bool = False,
        devices: set[torch.device] | None = None,
    ):
        # NOTE(pratik): this class assumes a zero-mean GP prior
        # Extract kernel information
        self.kernel_type = kernel_type
        self.kernel_config = KernelConfig(
            const_scaling=kernel_hparams.signal_variance,
            lengthscale=kernel_hparams.kernel_lengthscale,
        )
        self.noise_variance = kernel_hparams.noise_variance

        # Extract datasets
        self.Xtr = Xtr
        self.ytr = ytr
        self.Xtst = Xtst
        self.ytst = ytst

        # Extract information for posterior sampling
        self.num_posterior_samples = num_posterior_samples
        self.rf_config = rf_config
        (
            self.Xtr_prior_samples,
            self.Xtst_prior_samples,
        ) = self._get_approx_prior_samples()

        # Extract information for training
        self.distributed = distributed
        self.devices = devices

    def _get_approx_prior_samples(self):
        if self.num_posterior_samples > 0 and self.rf_config.num_features > 0:
            # Be careful is Xtst is None
            if self.Xtst is not None:
                X_in = torch.cat((self.Xtr, self.Xtst), dim=0)
            else:
                X_in = self.Xtr

            prior_samples = get_prior_samples(
                X_in,
                self.rf_config,
                self.kernel_config,
                self.kernel_type,
                self.noise_variance,
                self.num_posterior_samples,
                return_feature_weights=False,
            )

            # Make sure to return None for the test samples if Xtst is None
            if self.Xtst is not None:
                return (
                    prior_samples[: self.Xtr.shape[0]],
                    prior_samples[self.Xtr.shape[0] :],
                )
            else:
                return (prior_samples, None)
        return (None,) * 2

    def _get_linsys(self, use_full_kernel: bool):
        if self.Xtr_prior_samples is not None:
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
            kernel_config=self.kernel_config,
            use_full_kernel=use_full_kernel,
            distributed=self.distributed,
            devices=self.devices,
        )

    def _get_tst_kernel_linop(self):
        return _get_kernel_linop(
            self.Xtst,
            self.Xtr,
            kernel_type=self.kernel_type,
            kernel_config=self.kernel_config,
            distributed=self.distributed,
            devices=self.devices,
        )

    def _compute_nll(
        self,
        means: torch.Tensor,
        variances: torch.Tensor,
        locs: torch.Tensor,
        add_noise_variance: bool = True,
    ):
        if add_noise_variance:
            variances += self.noise_variance

        n = means.shape[0]
        log_variances = torch.log(variances)
        nll = 0.5 * (
            torch.sum((locs - means) ** 2 / variances)
            + torch.sum(log_variances)
            + n * torch.log(torch.tensor(2 * torch.pi))
        )
        return nll

    def _callback_fn(self, W: torch.Tensor, linsys: KernelLinSys, tst_kernel_linop):
        metrics_dict = {}

        # Compute train and test RMSE and R^2
        if linsys.use_full_kernel:
            train_mean = linsys.A @ W[:, 0]
            train_rmse = torch.sqrt(
                1 / self.ytr.shape[0] * torch.sum((self.ytr - train_mean) ** 2)
            )
            train_r2 = _get_r2(self.ytr, train_mean)
            metrics_dict.update(
                {
                    "train_rmse": train_rmse.cpu().item(),
                    "train_r2": train_r2.cpu().item(),
                }
            )

        test_mean = tst_kernel_linop @ W[:, 0]
        test_rmse = torch.sqrt(
            1 / self.ytst.shape[0] * torch.sum((self.ytst - test_mean) ** 2)
        )
        test_r2 = _get_r2(self.ytst, test_mean)
        metrics_dict.update(
            {
                "test_rmse": test_rmse.cpu().item(),
                "test_r2": test_r2.cpu().item(),
                "test_mean": test_mean.cpu().numpy(),
            }
        )

        # Compute variances and negative log likelihood using posterior samples
        if self.Xtst_prior_samples is not None:
            W_diff = _safe_unsqueeze(W[:, 0]) - W[:, 1:]
            test_posterior_samples = self.Xtst_prior_samples + tst_kernel_linop @ W_diff
            test_posterior_samples_mean = test_posterior_samples.mean(
                dim=1
            )  # Useful for sanity checking with test_mean
            test_posterior_samples_var = test_posterior_samples.var(dim=1)
            test_posterior_samples_nll = self._compute_nll(
                means=test_mean,  # Use the posterior mean without posterior sampling
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
        W_init: torch.Tensor | None = None,
        use_full_kernel: bool = True,
        eval_freq: int = 10,
        log_in_wandb: bool = False,
        wandb_init_kwargs: dict = {},
    ):
        linsys = self._get_linsys(use_full_kernel)
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
