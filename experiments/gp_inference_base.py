import argparse

# import os
# import pickle

# from scalable_gp_inference.gp_inference import GPInference

from experiments.constants import DATA_NAMES, EXPERIMENT_KERNELS
from experiments.utils import (
    device_type,
    dtype_type,
    none_or_str,
    # set_random_seed,
    # get_saved_gp_hparams,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Perform GP inference on a dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATA_NAMES,
        help="The name of the dataset to perform inference on",
    )
    parser.add_argument(
        "--kernel_type",
        type=str,
        choices=EXPERIMENT_KERNELS,
        help="The kernel type to use for the GP",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="The random seed to use",
    )
    parser.add_argument(
        "--devices",
        type=device_type,
        nargs="+",
        help="Devices to use: 'cpu' or GPU device ID (non-neg. integer)",
    )
    parser.add_argument(
        "--split_proportion",
        type=float,
        help="Proportion of data to use for testing",
    )
    parser.add_argument(
        "--split_shuffle",
        type=bool,
        help="Whether to shuffle the data before splitting",
    )
    parser.add_argument(
        "--standardize",
        type=bool,
        help="Whether to standardize the data",
    )
    parser.add_argument(
        "--dtype",
        type=dtype_type,
        help="Data type to use for optimization (e.g., 'float32', 'float64')",
    )
    parser.add_argument(
        "--num_posterior_samples",
        type=int,
        help="Number of posterior samples to draw from the GP",
    )
    parser.add_argument(
        "--num_random_features",
        type=int,
        help="Number of random features to use for approximation in GP inference",
    )
    parser.add_argument(
        "--use_full_kernel",
        type=bool,
        help="Whether to use the full kernel during inference",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        help="Frequency of evaluation during inference",
    )
    parser.add_argument(
        "--log_in_wandb",
        type=bool,
        help="Whether to log results in Weights & Biases",
    )
    parser.add_argument(
        "--opt_max_passes",
        type=int,
        help="Maximum number of passes for the optimizer",
    )
    parser.add_argument(
        "--opt_preconditioner",
        type=none_or_str,
        choices=["nystrom", "identity", "None"],
        help="Preconditioner to use for optimization -- SAP and PCG only",
    )
    parser.add_argument(
        "--opt_rank",
        type=int,
        default=None,
        help="Rank for the preconditioner -- nystrom only",
    )
    parser.add_argument(
        "--opt_damping",
        type=none_or_str,
        choices=["adaptive", "non_adaptive", "None"],
        help="Damping for the preconditioner -- nystrom only",
    )
    parser.add_argument(
        "--opt_blocks",
        type=int,
        default=None,
        help="Number of blocks for the optimizer -- SAP and SDD only",
    )
    parser.add_argument(
        "--opt_step_size_unscaled",
        type=float,
        default=None,
        help="Step size for the optimizer -- SDD only",
    )
    return parser.parse_args()
