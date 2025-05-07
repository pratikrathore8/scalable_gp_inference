import argparse
import time

import torch
from rlaopt.kernels import KernelConfig
from scalable_gp_inference.bayes_opt.configs import BayesOptConfig, TSConfig
from scalable_gp_inference.bayes_opt.core import BayesOpt

from experiments.constants import (
    BO_MAX_PASSES_PER_ITER,
    BO_MAX_ITERS,
    BO_KERNEL_TYPE,
    BO_KERNEL_CONST_SCALING,
    OPT_TYPES,
    OPT_ATOL,
    OPT_RTOL,
    OPT_RANK,
    OPT_DAMPING,
    OPT_SAP_PRECONDITIONERS,
    OPT_PCG_PRECONDITIONERS,
    OPT_SDD_STEP_SIZES_UNSCALED,
    LOGGING_USE_WANDB,
)
from experiments.utils import device_type, set_precision, set_random_seed

PRECISION = torch.float32

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run Bayesian optimization for a Matern-3/2 kernel for a given lengthscale, seed, and device"
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
    set_precision(PRECISION)

    # Set up parameters for kernel
    kernel_config = KernelConfig(const_scaling=BO_KERNEL_CONST_SCALING, lengthscale=args.lengthscale)
