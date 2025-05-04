from dataclasses import dataclass

from rlaopt.kernels import KernelConfig
import torch
from torch.distributions import Chi2


NU_MAP = {"matern12": 0.5, "matern32": 1.5, "matern52": 2.5}


def _get_safe_lengthscale(lengthscale: float | torch.Tensor) -> torch.Tensor:
    if isinstance(lengthscale, torch.Tensor):
        return lengthscale.unsqueeze(-1)
    return lengthscale


def _random_features(
    X: torch.Tensor, const_scaling: float, weights: dict
) -> torch.Tensor:
    scale_factor = (const_scaling * 2.0 / weights["Omega"].shape[1]) ** 0.5

    # Use in-place operations to reduce memory usage
    # Equivalent to result = scale_factor * torch.cos(X @ Omega + B)
    result = X @ weights["Omega"]
    result.add_(weights["B"])
    torch.cos(result, out=result)
    result.mul_(scale_factor)
    return result


@dataclass(kw_only=True, frozen=False)
class RFConfig:
    num_features: int
    regenerate: bool = True


class RandomFeatures:
    def __init__(
        self,
        kernel_config: KernelConfig,
        kernel_type: str,
        rf_config: RFConfig,
    ):
        self._check_kernel_type(kernel_type)
        self.kernel_config = kernel_config
        self.kernel_type = kernel_type
        self.rf_config = rf_config
        self.fixed_weights = None

    def _check_kernel_type(kernel_type: str):
        if kernel_type not in ["rbf", "matern12", "matern32", "matern52"]:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def _generate_weights(self, X: torch.Tensor):
        # NOTE(pratik): implicitly assumes kernel_type is rbf or matern
        num_features = self.rf_config.num_features
        Omega = torch.randn(X.shape[1], num_features, device=X.device, dtype=X.dtype)
        B = 2 * torch.pi * torch.rand(num_features, device=X.device, dtype=X.dtype)

        # Adjust Omega depending on the kernel
        safe_lengthscale = _get_safe_lengthscale(self.kernel_config.lengthscale)
        Omega /= safe_lengthscale
        if self.kernel_type in ["matern12", "matern32", "matern52"]:
            nu = NU_MAP[self.kernel_type]
            df = torch.tensor([2.0 * nu], device=X.device, dtype=X.dtype)
            # The sample method adds an extra dimension since df is a tensor,
            # so we need to squeeze it out
            u = Chi2(df).sample(sample_shape=(Omega.shape[1],)).squeeze(-1)
            Omega = torch.sqrt(df) * Omega / torch.sqrt(u)

        return dict(Omega=Omega, B=B)

    def get_random_features(self, X: torch.Tensor):
        # If we want fresh samples, regenerate the weights
        if self.rf_config.regenerate:
            weights = self._generate_weights(X)
        else:
            # Otherwise check if we already have fixed weights stored
            if self.fixed_weights is None:
                self.fixed_weights = self._generate_weights(X)
            weights = self.fixed_weights

        return _random_features(X, self.kernel_config.constant_scaling, weights)


def get_prior_samples(
    X: torch.Tensor,
    rf_obj: RandomFeatures,
    noise_variance: float,
    num_samples: int,
    return_feature_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    prior_samples = torch.empty(
        X.shape[0],
        num_samples,
        device=X.device,
        dtype=X.dtype,
    )

    if return_feature_weights:
        W = torch.empty(
            num_samples, rf_obj.rf_config.num_features, dtype=X.dtype, device=X.device
        )

    for i in range(num_samples):
        X_featurized = rf_obj.get_random_features(X)
        w = torch.randn(X_featurized.shape[1], device=X.device, dtype=X.dtype)
        prior_samples[:, i] = X_featurized @ w
        prior_samples[:, i] = prior_samples[:, i] + (
            noise_variance**0.5
        ) * torch.randn(X.shape[0], device=X.device, dtype=X.dtype)

        if return_feature_weights:
            W[i, :] = w.clone()  # Clone (for safety) since we delete w later

        # Free up memory
        del X_featurized, w
        torch.cuda.empty_cache()

    if return_feature_weights:
        return prior_samples, W
    return prior_samples, None
