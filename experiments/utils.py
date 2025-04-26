import argparse
import os
import pickle
import random

import numpy as np
import torch

# from rlaopt.preconditioners import IdentityConfig, NystromConfig
# from rlaopt.solvers import PCGConfig, SAPConfig, SAPAccelConfig
# from scalable_gp_inference.sdd_config import SDDConfig

from experiments.constants import (
    GP_TRAIN_SAVE_DIR,
    GP_TRAIN_SAVE_FILE_NAME,
    # EXPERIMENT_ATOL,
    # EXPERIMENT_RTOL,
)


def set_precision(precision):
    if precision == "float32":
        torch.set_default_dtype(torch.float32)
    elif precision == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError("Precision must be either 'float32' or 'float64'")


def set_random_seed(seed: int):
    """
    Set the random seed for reproducibility across NumPy, Python's random module,
    and PyTorch.

    This function ensures that the random number generation is
    consistent and reproducible by setting the same seed across different libraries.
    It also sets the seed for CUDA if a GPU is being used.

    Args:
        seed (int): The seed value to use for random number generation.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def device_type(value):
    """Custom type function for argparse to validate and format device argument"""
    if value.lower() == "cpu":
        return torch.device("cpu")

    try:
        gpu_id = int(value)
        if gpu_id < 0:
            raise argparse.ArgumentTypeError(
                "GPU device ID must be a non-negative integer"
            )

        if gpu_id >= torch.cuda.device_count():
            raise argparse.ArgumentTypeError(
                f"GPU device ID {gpu_id} is out of range. "
                f"Available devices: 0-{torch.cuda.device_count() - 1}"
            )

        return torch.device(f"cuda:{gpu_id}")
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Device must be 'cpu' or a non-negative integer"
        )


def dtype_type(value):
    """Custom type function for argparse to validate and format dtype argument"""
    if value.lower() == "float32":
        return torch.float32
    elif value.lower() == "float64":
        return torch.float64
    else:
        raise argparse.ArgumentTypeError("Data type must be 'float32' or 'float64'")


def none_or_str(value):
    if value == "None":
        return None
    return value


def get_solver_config(
    opt_type: str,
    max_passes: int,
    preconditioner: str,
    rank: int,
    regularization: float,
    damping: str,
    blocks: int,
    step_size_unscaled: float,
):
    raise NotImplementedError(
        "This function is not implemented. Please implement "
        "the function to get the solver configuration."
    )


def get_gp_hparams_save_file_dir(
    dataset_name: str,
    kernel_type: str,
    seed: int,
):
    """
    Generate a directory name for saving GPHparams based on the dataset name,
    kernel type, and random seed.

    Args:
        dataset_name (str): The name of the dataset.
        kernel_type (str): The type of kernel used.
        seed (int): The random seed used for training.

    Returns:
        str: The generated directory name.
    """
    return os.path.join(
        GP_TRAIN_SAVE_DIR,
        dataset_name,
        kernel_type,
        f"seed_{seed}",
    )


def get_saved_gp_hparams(
    dataset_name: str,
    kernel_type: str,
    seed: int,
):
    """
    Load saved GP hyperparameters from a file.

    Args:
        dataset_name (str): The name of the dataset.
        kernel_type (str): The type of kernel used.
        seed (int): The random seed used for training.

    Returns:
        dict: The loaded GP hyperparameters.
    """
    save_dir = get_gp_hparams_save_file_dir(dataset_name, kernel_type, seed)
    save_file = os.path.join(save_dir, GP_TRAIN_SAVE_FILE_NAME)
    if not os.path.exists(save_file):
        raise FileNotFoundError(f"GP hyperparameters file not found: {save_file}")
    with open(save_file, "rb") as f:
        gp_hparams = pickle.load(f)

    return gp_hparams
