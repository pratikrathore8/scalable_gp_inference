from rlaopt.kernels import KernelConfig
import torch
from torch.distributions import Chi2

from .tanimoto_kernel import TanimotoKernelConfig


def _get_safe_lengthscale(lengthscale: float | torch.Tensor) -> torch.Tensor:
    if isinstance(lengthscale, torch.Tensor):
        return lengthscale.unsqueeze(-1)
    return lengthscale


def _random_features(
    X: torch.Tensor, num_features: int, const_scaling: float, Omega: torch.Tensor
) -> torch.Tensor:
    B = 2 * torch.pi * torch.rand(num_features, device=X.device, dtype=X.dtype)
    scale_factor = (const_scaling * 2.0 / num_features) ** 0.5

    # Use in-place operations to reduce memory usage
    # Equivalent to result = scale_factor * torch.cos(X @ Omega + B)
    result = X @ Omega
    result.add_(B)
    torch.cos(result, out=result)
    result.mul_(scale_factor)
    return result


def _rbf_random_features(
    X: torch.Tensor, num_features: int, kernel_config: KernelConfig
) -> torch.Tensor:
    safe_lengthscale = _get_safe_lengthscale(kernel_config.lengthscale)
    Omega = (
        torch.randn(X.shape[1], num_features, device=X.device, dtype=X.dtype)
        / safe_lengthscale
    )
    return _random_features(X, num_features, kernel_config.const_scaling, Omega)


def _matern_random_features(
    X: torch.Tensor,
    num_features: int,
    kernel_config: KernelConfig,
    nu: float,
) -> torch.Tensor:
    safe_lengthscale = _get_safe_lengthscale(kernel_config.lengthscale)
    # Construction is using the multivariate t-distribution
    # (Figure 1 in https://mlg.eng.cam.ac.uk/adrian/geometry.pdf
    # -- this requires care, since there are typos in the expression).
    # Sampling from the multivariate t-distribution can be performed
    # using the normal and chi-square distributions.
    # See https://en.wikipedia.org/wiki/Multivariate_t-distribution for details.
    Y = (
        torch.randn(X.shape[1], num_features, device=X.device, dtype=X.dtype)
        / safe_lengthscale
    )
    df = torch.tensor([2.0 * nu], device=X.device, dtype=X.dtype)
    # The sample method adds an extra dimension since df is a tensor,
    # so we need to squeeze it out
    u = Chi2(df).sample(sample_shape=(num_features,)).squeeze(-1)
    Omega = torch.sqrt(df) * Y / torch.sqrt(u)
    return _random_features(X, num_features, kernel_config.const_scaling, Omega)


def _tanimoto_random_features(
    X: torch.Tensor,
    num_features: int,
    kernel_config: TanimotoKernelConfig,
    modulo_value: int,
) -> torch.Tensor:
    """
    Compute random features for Tanimoto kernel approximation
    with vectorized operations.

    Args:
        x: Input tensor of shape (batch_size, D)
        num_features: Number of random features to generate
        modulo_value: Modulo value for feature hashing

    Returns:
        Random features tensor of shape (batch_size, n_features)
    """
    batch_size, D = X.shape
    M = num_features
    device = X.device

    # Generate random parameters
    r = -torch.log(torch.rand(M, D, device=device)) - torch.log(
        torch.rand(M, D, device=device)
    )
    c = -torch.log(torch.rand(M, D, device=device)) - torch.log(
        torch.rand(M, D, device=device)
    )
    xi = (
        torch.randint(0, 2, (M, D, modulo_value), device=device) * 2 - 1
    )  # Rademacher distribution
    beta = torch.rand(M, D, device=device)

    # Process each feature independently for better memory management
    features = torch.zeros(batch_size, M, device=device)

    for m in range(M):
        # Compute log-space transformation for all samples
        t = torch.floor(
            torch.log(X) / r[m].unsqueeze(0) + beta[m].unsqueeze(0)
        )  # (batch_size, D)
        ln_y = r[m].unsqueeze(0) * (t - beta[m].unsqueeze(0))  # (batch_size, D)
        ln_a = (
            torch.log(c[m].unsqueeze(0)) - ln_y - r[m].unsqueeze(0)
        )  # (batch_size, D)

        # Find argmin for each sample
        a_argmin = torch.argmin(ln_a, dim=1)  # (batch_size,)

        # Get corresponding t values
        batch_indices = torch.arange(batch_size, device=device)
        t_selected = t[batch_indices, a_argmin].long() % modulo_value  # (batch_size,)

        # Select features from xi
        features[:, m] = xi[m, a_argmin, t_selected]

    scale_factor = (kernel_config.const_scaling / M) ** 0.5

    return scale_factor * features


def get_random_features(
    X: torch.Tensor,
    num_features: int,
    kernel_config: KernelConfig,
    kernel_type: str,
) -> torch.Tensor:
    if kernel_type == "rbf":
        return _rbf_random_features(X, num_features, kernel_config)
    elif kernel_type == "matern12":
        return _matern_random_features(X, num_features, kernel_config, nu=0.5)
    elif kernel_type == "matern32":
        return _matern_random_features(X, num_features, kernel_config, nu=1.5)
    elif kernel_type == "matern52":
        return _matern_random_features(X, num_features, kernel_config, nu=2.5)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
