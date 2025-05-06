from dataclasses import dataclass

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel


SUBSAMPLE_BATCH_SIZE = 5 * 10**7  # Adjust based on available memory


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
        # Check types
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

        # Check positivity
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

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> "GPHparams":
        """Move GPHparams to a specified device and/or convert to a specified dtype."""
        # If kernel_lengthscale is not a tensor,
        # then there's nothing to convert, so return self
        if not isinstance(self.kernel_lengthscale, torch.Tensor):
            return self

        # Handle tensor conversion
        lengthscale = self.kernel_lengthscale.to(device=device, dtype=dtype)

        # Only create a new instance if the tensor actually changed
        if lengthscale is self.kernel_lengthscale:
            return self

        return GPHparams(
            signal_variance=self.signal_variance,
            kernel_lengthscale=lengthscale,
            noise_variance=self.noise_variance,
        )

    def to_dict(self) -> dict[str, float | list[float]]:
        """Convert GPHparams to a dictionary."""
        # Check if kernel_lengthscale is a tensor
        if isinstance(self.kernel_lengthscale, torch.Tensor):
            # Convert to list if it's a tensor
            kernel_lengthscale = self.kernel_lengthscale.tolist()
        else:
            kernel_lengthscale = self.kernel_lengthscale

        return {
            "signal_variance": self.signal_variance,
            "kernel_lengthscale": kernel_lengthscale,
            "noise_variance": self.noise_variance,
        }

    def __add__(self, other: "GPHparams") -> "GPHparams":
        """Add two GPHparams instances, returning a new instance."""
        if not isinstance(other, GPHparams):
            raise TypeError(f"Cannot add type {type(other).__name__} to GPHparams")

        return GPHparams(
            signal_variance=self.signal_variance + other.signal_variance,
            kernel_lengthscale=self.kernel_lengthscale + other.kernel_lengthscale,
            noise_variance=self.noise_variance + other.noise_variance,
        )

    def __iadd__(self, other: "GPHparams") -> "GPHparams":
        """'In-place' addition of two GPHparams instances.

        This is not a standard in-place operation, but rather a
        convenience method to allow for a more intuitive syntax.
        """
        return self + other

    def __truediv__(self, scalar: float | int) -> "GPHparams":
        """Divide GPHparams by a scalar value."""
        if not isinstance(scalar, (float, int)):
            raise TypeError(f"Cannot divide GPHparams by type {type(scalar).__name__}")

        if scalar == 0:
            raise ZeroDivisionError("Cannot divide GPHparams by zero")

        # Divide each parameter by the scalar
        return GPHparams(
            signal_variance=self.signal_variance / scalar,
            kernel_lengthscale=self.kernel_lengthscale / scalar,
            noise_variance=self.noise_variance / scalar,
        )


def _train_exact_gp(
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    kernel_type: str,
    opt_class: torch.optim.Optimizer,
    opt_hparams: dict,
    training_iters: int,
) -> GPHparams:
    # Initialize likelihood and model
    likelihood = GaussianLikelihood()

    # Get base kernel
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

    # Find optimal hyperparameters
    model.train()
    likelihood.train()

    # Initialize optimizer (typically Adam)
    # Includes GaussianLikelihood parameters
    optimizer = opt_class(model.parameters(), **opt_hparams)

    # "Loss" for GPs is the marginal log likelihood
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
        ),  # Get rid of the extra dimension
        noise_variance=likelihood.noise.item(),
    )


def _get_subsample_centroid(
    Xtr: torch.Tensor, ytr: torch.Tensor, subsample_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # Randomly sample a point from the training set
    idx = torch.randint(0, Xtr.shape[0], (1,))
    centroid = Xtr[idx].view(1, -1)

    # Process in batches and maintain the k closest points seen so far
    batch_size = SUBSAMPLE_BATCH_SIZE  # Adjust based on available memory
    num_batches = (Xtr.shape[0] + batch_size - 1) // batch_size

    # Initialize with a very large finite value
    max_value = torch.finfo(Xtr.dtype).max
    closest_distances = torch.full(
        (subsample_size,), max_value, device=Xtr.device, dtype=Xtr.dtype
    )
    closest_indices = torch.zeros(subsample_size, dtype=torch.long, device=Xtr.device)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, Xtr.shape[0])

        # Compute distances for this batch using cdist
        batch_distances = torch.cdist(Xtr[start_idx:end_idx], centroid).squeeze(-1)

        # Add batch offset to make indices global
        batch_indices = torch.arange(start_idx, end_idx, device=Xtr.device)

        # Combine current batch with the closest points found so far
        combined_distances = torch.cat([closest_distances, batch_distances])
        combined_indices = torch.cat([closest_indices, batch_indices])

        # Get the top-k closest points from the combined set
        topk_dists, topk_idx = torch.topk(
            combined_distances, k=subsample_size, largest=False, sorted=False
        )

        # Update our running closest points
        closest_distances = topk_dists
        closest_indices = combined_indices[topk_idx]

    # Return the final subset
    return Xtr[closest_indices], ytr[closest_indices]


def train_exact_gp_subsampled(
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    kernel_type: str,
    opt_class: torch.optim.Optimizer,
    opt_hparams: dict,
    training_iters: int,
    subsample_size: int,
    num_trials: int,
) -> GPHparams:
    gp_hparams = None

    # Train a GP on subsamples of the training data
    for i in range(num_trials):
        # Get a random, centroid-based subsample of the training data
        Xtr_subsampled, ytr_subsampled = _get_subsample_centroid(
            Xtr, ytr, subsample_size
        )

        # Train the GP on the subsample
        if gp_hparams is None:
            gp_hparams = _train_exact_gp(
                Xtr_subsampled,
                ytr_subsampled,
                kernel_type,
                opt_class,
                opt_hparams,
                training_iters,
            )
        else:
            gp_hparams_i = _train_exact_gp(
                Xtr_subsampled,
                ytr_subsampled,
                kernel_type,
                opt_class,
                opt_hparams,
                training_iters,
            )
            gp_hparams += gp_hparams_i

    # Return the hyperparameters after averaging over the trials
    return gp_hparams / num_trials
