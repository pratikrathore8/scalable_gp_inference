import argparse
import os  # noqa: F401
import pickle  # noqa: F401

from scalable_gp_inference.gp_inference import GPInference  # noqa: F401

from experiments.constants import DATA_NAMES, EXPERIMENT_KERNELS
from experiments.utils import (
    device_type,
    dtype_type,
    # set_random_seed,  # noqa: F401
    # get_saved_gp_hparams,  # noqa: F401
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
        "--eval-freq",
        type=int,
        help="Frequency of evaluation during inference",
    )
    parser.add_argument(
        "--log_in_wandb",
        type=bool,
        help="Whether to log results in Weights & Biases",
    )
    parser.add_argument()
    return parser.parse_args()
