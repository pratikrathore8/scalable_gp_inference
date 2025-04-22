from dataclasses import dataclass

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel


class _ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, base_kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


@dataclass(kw_only=True, frozen=True)
class GPHparams:
    signal_variance: float
    kernel_lengthscale: float | torch.Tensor
    noise_variance: float

    def __post_init__(self):
        # check types
        if not isinstance(self.signal_variance, float):
            raise TypeError(
                f"signal_variance is of type {type(self.signal_variance).__name__}, "
                "but expected type float"
            )
        if not isinstance(self.kernel_lengthscale, (float, torch.Tensor)):
            raise TypeError(
                "kernel_lengthscale is of "
                f"type {type(self.kernel_lengthscale).__name__}, "
                "but expected type float or torch.Tensor"
            )
        if not isinstance(self.noise_variance, float):
            raise TypeError(
                f"noise_variance is of type {type(self.noise_variance).__name__}, "
                "but expected type float"
            )

        # check positivity
        if self.signal_variance < 0:
            raise ValueError("signal_variance must be non-negative")
        if isinstance(self.kernel_lengthscale, float) and self.kernel_lengthscale <= 0:
            raise ValueError("kernel_lengthscale must be positive")
        elif (
            isinstance(self.kernel_lengthscale, torch.Tensor)
            and (self.kernel_lengthscale <= 0).any()
        ):
            raise ValueError("kernel_lengthscale must be positive")
        if self.noise_variance < 0:
            raise ValueError("noise_variance must be non-negative")

    def __add__(self, other: "GPHparams") -> "GPHparams":
        """Add two GPHparams instances, returning a new instance."""
        if not isinstance(other, GPHparams):
            raise TypeError(f"Cannot add type {type(other).__name__} to GPHparams")

        return GPHparams(
            signal_variance=self.signal_variance + other.signal_variance,
            kernel_lengthscale=self.kernel_lengthscale + other.kernel_lengthscale,
            noise_variance=self.noise_variance + other.noise_variance,
        )

    def __truediv__(self, scalar: float | int) -> "GPHparams":
        """Divide GPHparams by a scalar value."""
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Cannot divide GPHparams by type {type(scalar).__name__}")

        if scalar == 0:
            raise ZeroDivisionError("Cannot divide GPHparams by zero")

        # Divide each parameter by the scalar
        return GPHparams(
            signal_variance=self.signal_variance / scalar,
            kernel_lengthscale=self.kernel_lengthscale / scalar,
            noise_variance=self.noise_variance / scalar,
        )


def train_exact_gp(
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    kernel_type: str,
    opt_hparams: dict,
    training_iters: int,
) -> GPHparams:
    # initialize likelihood and model
    likelihood = GaussianLikelihood()

    # get base kernel
    if kernel_type == "rbf":
        base_kernel = RBFKernel(ard_num_dims=Xtr.shape[1])
    elif kernel_type == "matern12":
        base_kernel = MaternKernel(nu=0.5, ard_num_dims=Xtr.shape[1])
    elif kernel_type == "matern32":
        base_kernel = MaternKernel(nu=1.5, ard_num_dims=Xtr.shape[1])
    elif kernel_type == "matern52":
        base_kernel = MaternKernel(nu=2.5, ard_num_dims=Xtr.shape[1])
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    model = _ExactGPModel(Xtr, ytr, likelihood, base_kernel)
    model = model.to(Xtr.device)
    likelihood = likelihood.to(Xtr.device)

    # find optimal hyperparameters
    model.train()
    likelihood.train()

    # use Adam optimizer
    # includes GaussianLikelihood parameters
    optimizer = torch.optim.Adam(model.parameters(), **opt_hparams)

    # "loss" for GPs is the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iters):
        optimizer.zero_grad()
        output = model(Xtr)
        loss = -mll(output, ytr)
        loss.backward()
        optimizer.step()

    return GPHparams(
        signal_variance=model.covar_module.outputscale.item(),
        kernel_lengthscale=model.covar_module.base_kernel.lengthscale.detach().squeeze(
            0
        ),  # get rid of the extra dimension
        noise_variance=likelihood.noise.item(),
    )
