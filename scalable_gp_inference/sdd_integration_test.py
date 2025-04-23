import torch
from scalable_gp_inference.kernel_linsys import KernelLinSys
from scalable_gp_inference.sdd_config import SDDConfig

from rlaopt.kernels import KernelConfig
from rlaopt.preconditioners import NystromConfig
from rlaopt.solvers import PCGConfig, SAPConfig, SAPAccelConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n = 100000
d = 3
k = 10
max_iters = 1000

# Large regularization to check that SDD actually converges when it should
reg = 1e-2 * n

kernel_type = "rbf"
kernel_lengthscale = torch.tensor([1.0, 2.0, 3.0], device=device)
residual_tracking_idx = None
distributed = False
devices = None
kernel_config = KernelConfig(const_scaling=1.0, lengthscale=kernel_lengthscale)

X = torch.randn(n, d, device=device)
B = torch.randn(n, k, device=device)

kernel_linsys = KernelLinSys(
    X, B, reg, kernel_type, kernel_config, residual_tracking_idx, distributed, devices
)

nystrom_config = NystromConfig(rank=100, rho=reg, damping_mode="adaptive")
accel_config = SAPAccelConfig(mu=reg / 10000, nu=10.0)
solver_config = SAPConfig(
    precond_config=nystrom_config,
    max_iters=max_iters,
    atol=1e-6,
    rtol=1e-6,
    blk_sz=n // 10,
    accel=False,
    accel_config=None,
    device=device,
)

print("Running ASkotch")
_, log_asko = kernel_linsys.solve(
    solver_config=solver_config, W_init=torch.zeros_like(B), log_in_wandb=False
)

final_log_entry_asko = log_asko[list(log_asko.keys())[-1]]
print("Final log entry key:", list(log_asko.keys())[-1])
print("Final log entry:", final_log_entry_asko)

# print("Running PCG")
# solver_config = PCGConfig(
#     precond_config=nystrom_config,
#     max_iters=1000,
#     atol=1e-6,
#     rtol=1e-6,
#     device=device
# )

# solution, log = kernel_linsys.solve(
#     solver_config=solver_config,
#     W_init=torch.zeros_like(B),
#     log_in_wandb=False
# )

# final_log_entry = log[list(log.keys())[-1]]
# print("Final log entry key:", list(log.keys())[-1])
# print("Final log entry:", final_log_entry)

print("Running SDD")

solver_config = SDDConfig(
    blk_size=n // 10,
    step_size=0.1 / n,
    m=0.9,
    theta=100 / max_iters,
    max_iters=max_iters,
    device=device,
)

_, log_sdd = kernel_linsys.solve(
    solver_config=solver_config, W_init=torch.zeros_like(B), log_in_wandb=False
)

final_log_entry_sdd = log_sdd[list(log_sdd.keys())[-1]]
print("Final log entry key:", list(log_sdd.keys())[-1])
print("Final log entry:", final_log_entry_sdd)
