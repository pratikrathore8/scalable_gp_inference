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


# def _tanimoto_random_features(
#     X: torch.Tensor,
#     num_features: int,
#     kernel_config: TanimotoKernelConfig,
#     modulo_value: int,
# ) -> torch.Tensor:
#     """
#     Compute random features for Tanimoto kernel approximation
#     with vectorized operations.

#     Args:
#         x: Input tensor of shape (batch_size, D)
#         num_features: Number of random features to generate
#         modulo_value: Modulo value for feature hashing

#     Returns:
#         Random features tensor of shape (batch_size, n_features)
#     """
#     batch_size, D = X.shape
#     M = num_features
#     device = X.device

#     # Generate random parameters
#     r = -torch.log(torch.rand(M, D, device=device)) - torch.log(
#         torch.rand(M, D, device=device)
#     )
#     c = -torch.log(torch.rand(M, D, device=device)) - torch.log(
#         torch.rand(M, D, device=device)
#     )
#     xi = (
#         torch.randint(0, 2, (M, D, modulo_value), device=device) * 2 - 1
#     )  # Rademacher distribution
#     beta = torch.rand(M, D, device=device)

#     # Process each feature independently for better memory management
#     features = torch.zeros(batch_size, M, device=device)

#     for m in range(M):
#         # Compute log-space transformation for all samples
#         t = torch.floor(
#             torch.log(X) / r[m].unsqueeze(0) + beta[m].unsqueeze(0)
#         )  # (batch_size, D)
#         ln_y = r[m].unsqueeze(0) * (t - beta[m].unsqueeze(0))  # (batch_size, D)
#         ln_a = (
#             torch.log(c[m].unsqueeze(0)) - ln_y - r[m].unsqueeze(0)
#         )  # (batch_size, D)

#         # Find argmin for each sample
#         a_argmin = torch.argmin(ln_a, dim=1)  # (batch_size,)

#         # Get corresponding t values
#         batch_indices = torch.arange(batch_size, device=device)
#         t_selected = t[batch_indices, a_argmin].long() % modulo_value  # (batch_size,)

#         # Select features from xi
#         features[:, m] = xi[m, a_argmin, t_selected]

#     scale_factor = (kernel_config.const_scaling / M) ** 0.5

#     return scale_factor * features


def _tanimoto_random_features(
    X: torch.Tensor,
    num_features: int,
    kernel_config: TanimotoKernelConfig,
    modulo_value: int,
    feature_batch_size: int = 100,  # Process this many features at once
) -> torch.Tensor:
    """
    Compute random features for Tanimoto kernel approximation
      with true batched processing.
    Processes multiple features simultaneously for better efficiency.

    Args:
        X: Input tensor of shape (batch_size, D)
        num_features: Number of random features to generate
        kernel_config: Configuration for the kernel
        modulo_value: Modulo value for feature hashing
        feature_batch_size: Number of features to process in each batch

    Returns:
        Random features tensor of shape (batch_size, num_features)
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
    xi = torch.randint(0, 2, (M, D, modulo_value), device=device) * 2 - 1
    beta = torch.rand(M, D, device=device)

    # Calculate log(X) once for all batches
    log_X = torch.log(X)  # (batch_size, D)

    # Initialize output tensor
    features = torch.zeros(batch_size, M, device=device)

    # Process features in batches
    for m_start in range(0, M, feature_batch_size):
        m_end = min(m_start + feature_batch_size, M)
        current_batch_size = m_end - m_start

        # Select parameters for current batch of features
        r_batch = r[m_start:m_end]  # (current_batch_size, D)
        c_batch = c[m_start:m_end]  # (current_batch_size, D)
        beta_batch = beta[m_start:m_end]  # (current_batch_size, D)
        xi_batch = xi[m_start:m_end]  # (current_batch_size, D, modulo_value)

        # Reshape for broadcasting
        r_batch_expanded = r_batch.unsqueeze(1)  # (current_batch_size, 1, D)
        beta_batch_expanded = beta_batch.unsqueeze(1)  # (current_batch_size, 1, D)
        c_batch_expanded = c_batch.unsqueeze(1)  # (current_batch_size, 1, D)
        log_X_expanded = log_X.unsqueeze(0)  # (1, batch_size, D)

        # Compute transformations for all samples and features in this batch at once
        # Shape: (current_batch_size, batch_size, D)
        t_batch = torch.floor(log_X_expanded / r_batch_expanded + beta_batch_expanded)
        ln_y_batch = r_batch_expanded * (t_batch - beta_batch_expanded)
        ln_a_batch = torch.log(c_batch_expanded) - ln_y_batch - r_batch_expanded

        # Find argmin for each feature and sample
        # Shape: (current_batch_size, batch_size)
        a_argmin_batch = torch.argmin(ln_a_batch, dim=2)

        # Reshape indices for vectorized operation
        feature_indices = torch.arange(current_batch_size, device=a_argmin_batch.device)
        batch_indices = torch.arange(batch_size, device=a_argmin_batch.device)

        # Get all t values at once
        t_values = (
            t_batch[
                feature_indices.unsqueeze(1), batch_indices.unsqueeze(0), a_argmin_batch
            ].long()
            % modulo_value
        )  # [feature_count, batch_size]

        # Prepare indices for gathering from xi_batch
        feature_indices_expanded = (
            feature_indices.view(-1, 1).expand(-1, batch_size).reshape(-1)
        )
        arg_indices_flattened = a_argmin_batch.reshape(-1)
        t_values_flattened = t_values.reshape(-1)

        # Gather the values from xi_batch
        xi_values = xi_batch[
            feature_indices_expanded, arg_indices_flattened, t_values_flattened
        ]

        # Reshape back to original dimensions and assign to features
        features[:, m_start:m_end] = xi_values.reshape(
            current_batch_size, batch_size
        ).transpose(0, 1)

        # Clean up large intermediate tensors
        del t_batch, ln_y_batch, ln_a_batch, a_argmin_batch

    # Apply scaling
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
