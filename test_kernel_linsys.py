import torch

from rlaopt.kernels import KernelConfig
from rlaopt.solvers import PCGConfig, SAPConfig, SAPAccelConfig  # noqa: F401
from rlaopt.preconditioners import NystromConfig

from scalable_gp_inference.kernel_linsys import KernelLinSys

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)

    device = torch.device("cuda:1")

    n = 10**7
    d = 30
    k = 1
    reg = 1e-6
    kernel_type = "rbf"
    kernel_lengthscale = torch.ones(d, device=device)
    kernel_config = KernelConfig(const_scaling=1.0, lengthscale=kernel_lengthscale)
    use_full_kernel = False
    residual_tracking_idx = None
    distributed = True
    devices = set(
        [torch.device("cuda:1"), torch.device("cuda:3"), torch.device("cuda:4")]
    )

    X = torch.randn(n, d, device=device)
    B = torch.randn(n, k, device=device)

    kernel_linsys = KernelLinSys(
        X,
        B,
        reg,
        kernel_type,
        kernel_config,
        use_full_kernel,
        residual_tracking_idx,
        distributed,
        devices,
    )

    nystrom_config = NystromConfig(rank=100, rho=reg, damping_mode="adaptive")
    accel_config = SAPAccelConfig(mu=reg, nu=100.0)
    solver_config = SAPConfig(
        precond_config=nystrom_config,
        max_iters=30,
        atol=1e-6,
        rtol=1e-6,
        blk_sz=n // 100,
        accel_config=accel_config,
        device=device,
    )
    # solver_config = PCGConfig(
    #     precond_config=nystrom_config,
    #     max_iters=1000,
    #     atol=1e-6,
    #     rtol=1e-6,
    #     device=device,
    # )

    solution, log = kernel_linsys.solve(
        solver_config=solver_config,
        W_init=torch.zeros_like(B),
        callback_freq=100,
        log_in_wandb=True,
        wandb_init_kwargs={"project": "test_krr_linsys_class"},
    )

    final_log_entry = log[list(log.keys())[-1]]
    print("Final log entry key:", list(log.keys())[-1])
    print("Final log entry:", final_log_entry)
