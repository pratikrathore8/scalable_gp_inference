import argparse
import time

import torch
from rlaopt.kernels import KernelConfig
from scalable_gp_inference.bayes_opt.configs import BayesOptConfig, TSConfig
from scalable_gp_inference.bayes_opt.core import BayesOpt
import wandb

from experiments.constants import (
    BO_MAX_PASSES_PER_ITER,
    BO_MAX_ITERS,
    BO_NOISE_VARIANCE,
    BO_OPT_NUM_BLOCKS,
    BO_PRECISION,
    BO_KERNEL_TYPE,
    BO_KERNEL_CONST_SCALING,
    OPT_TYPES,
    OPT_RANK,
    OPT_DAMPING,
    OPT_SAP_PRECONDITIONERS,
    OPT_PCG_PRECONDITIONERS,
    OPT_SDD_STEP_SIZES_UNSCALED,
    LOGGING_USE_WANDB,
)
from experiments.utils import (
    device_type,
    get_solver_config,
    set_precision,
    set_random_seed,
)


OPT_PRECONDITIONERS_DICT = {
    "pcg": OPT_PCG_PRECONDITIONERS,
    "sap": OPT_SAP_PRECONDITIONERS,
}


def get_bo_obj(
    bo_config: BayesOptConfig, device: torch.device, dtype: torch.dtype
) -> BayesOpt:
    return BayesOpt(
        bo_config=bo_config,
        device=device,
        dtype=dtype,
    )


def get_solver_config_kwargs(
    opt_types: str,
    max_passes: int,
    opt_preconditioners_dict: dict,
    rank: int,
    regularization: float,
    damping: str,
    blocks: int,
    step_sizes_unscaled: float,
    device: torch.device,
) -> list[dict]:
    # Loop over opt_types to get list of solver config kwargs (don't put in ntr though)
    solver_config_kwargs = []
    for opt_type in opt_types:
        if opt_type == "sdd":
            # Loop over the step sizes
            for step_size_unscaled in step_sizes_unscaled:
                solver_config_kwargs.append(
                    {
                        "opt_type": "sdd",
                        "max_passes": max_passes,
                        "preconditioner": None,
                        "rank": None,
                        regularization: None,
                        "damping": None,
                        "blocks": blocks,
                        "step_size_unscaled": step_size_unscaled,
                        "device": device,
                    }
                )
        else:
            # Handle pcg and sap by looping over preconditioners
            for preconditioner in opt_preconditioners_dict[opt_type]:
                solver_config_kwargs.append(
                    {
                        "opt_type": opt_type,
                        "max_passes": max_passes,
                        "preconditioner": preconditioner,
                        "rank": rank,
                        "regularization": regularization,
                        "damping": damping,
                        "blocks": blocks,
                        "step_size_unscaled": None,
                        "device": device,
                    }
                )
    return solver_config_kwargs


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run Bayesian optimization for a Matern-3/2 kernel for a "
        "given lengthscale, seed, and device"
    )
    parser.add_argument(
        "--lengthscale",
        type=float,
        help="The lengthscale parameter for the Matern-3/2 kernel",
    )
    parser.add_argument("--seed", type=int, help="The random seed to use")
    parser.add_argument("--device", type=device_type, help="Device ID")
    args = parser.parse_args()

    # Set random seed for reproducibility
    set_random_seed(args.seed)

    # Set precision for training
    set_precision(BO_PRECISION)

    # Set up parameters for kernel
    kernel_config = KernelConfig(
        const_scaling=BO_KERNEL_CONST_SCALING, lengthscale=args.lengthscale
    )

    # Get config for Bayesian optimization
    bo_config = BayesOptConfig(
        kernel_config=kernel_config,
        kernel_type=BO_KERNEL_TYPE,
        noise_variance=BO_NOISE_VARIANCE,
    )

    # Get configs for Thompson sampling
    ts_config_uniform = TSConfig(exp_method="uniform")
    ts_config_nearby = TSConfig(exp_method="nearby")

    # Run Bayesian optimization with uniform exploration
    bo_uniform = get_bo_obj(bo_config=bo_config, device=args.device, dtype=BO_PRECISION)
    if LOGGING_USE_WANDB:
        wandb.init(
            project="bayesopt_lengthscale_{args.lengthscale}",
            config={
                "max_passes_per_iter": BO_MAX_PASSES_PER_ITER,
                "max_iters": BO_MAX_ITERS,
                "bo_config": bo_config.to_dict(),
                "ts_config": ts_config_uniform.to_dict(),
                "solver_config": None,
                "solver_name": None,
            },
        )
    for _ in range(BO_MAX_ITERS):
        ts = time.time()
        bo_uniform.step(ts_config=ts_config_uniform, krr_solver_config=None)
        te = time.time()
        if LOGGING_USE_WANDB:
            wandb.log(
                {
                    "iter_time": te - ts,
                    "num_acquisitions": len(bo_uniform.bo_state),
                    "fn_max": bo_uniform.bo_state.fn_max,
                    "fn_argmax": bo_uniform.bo_state.fn_argmax,
                }
            )

    # Run Bayesian optimization with nearby exploration
    # In this case, we actually use the PCG/SAP/SDD optimizers
    bo_nearby = get_bo_obj(bo_config=bo_config, device=args.device)
    solver_config_kwargs = get_solver_config_kwargs(
        opt_types=OPT_TYPES,
        max_passes=BO_MAX_PASSES_PER_ITER,
        opt_preconditioners_dict=OPT_PRECONDITIONERS_DICT,
        rank=OPT_RANK,
        regularization=BO_NOISE_VARIANCE,
        damping=OPT_DAMPING,
        blocks=BO_OPT_NUM_BLOCKS,
        step_sizes_unscaled=OPT_SDD_STEP_SIZES_UNSCALED,
        device=args.device,
    )
