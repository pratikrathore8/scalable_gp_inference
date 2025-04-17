from typing import Union
import torch
from torch.distributions import Chi2


def _get_safe_lengthscale(lengthscale: Union[float, torch.Tensor]) -> torch.Tensor:
    if isinstance(lengthscale, torch.Tensor):
        return lengthscale.unsqueeze(-1)
    return lengthscale


def _random_features(
    X: torch.Tensor, num_features: int, Omega: torch.Tensor
) -> torch.Tensor:
    B = 2 * torch.pi * torch.rand(num_features, device=X.device, dtype=X.dtype)
    scale_factor = (2.0 / num_features) ** 0.5
    return scale_factor * torch.cos(X @ Omega + B)


def rbf_random_features(
    X: torch.Tensor, lengthscale: Union[float, torch.Tensor], num_features: int
) -> torch.Tensor:
    safe_lengthscale = _get_safe_lengthscale(lengthscale)
    Omega = (
        torch.randn(X.shape[1], num_features, device=X.device, dtype=X.dtype)
        / safe_lengthscale
    )
    return _random_features(X, num_features, Omega)


def matern_random_features(
    X: torch.Tensor,
    lengthscale: Union[float, torch.Tensor],
    num_features: int,
    nu: float,
) -> torch.Tensor:
    safe_lengthscale = _get_safe_lengthscale(lengthscale)
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
    return _random_features(X, num_features, Omega)
