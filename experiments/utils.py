import argparse
import os
import pickle
import random

import numpy as np
import torch

from rlaopt.preconditioners import IdentityConfig, NystromConfig
from rlaopt.solvers import PCGConfig, SAPConfig, SAPAccelConfig
from scalable_gp_inference.random_features import RFConfig
from scalable_gp_inference.sdd_config import SDDConfig

from experiments.data_processing.load_torch import LOADERS
from experiments.data_processing.dockstring_params import (
    DOCKSTRING_DATASET_HPARAMS,
    dockstring_hparams_to_gphparams,
)
from experiments.constants import (
    GP_TRAIN_SAVE_DIR,
    GP_TRAIN_SAVE_FILE_NAME,
    OPT_ATOL,
    OPT_RTOL,
    OPT_SDD_MOMENTUM,
)


def load_dataset(args, device: torch.device):
    """
    Load the dataset based on the provided arguments.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    load_fn = LOADERS[args.dataset]
    if args.dataset in DOCKSTRING_DATASET_HPARAMS:
        dataset = load_fn(
            standardize=args.standardize,
            dtype=args.dtype,
            device=device,
        )
    else:
        dataset = load_fn(
            split_proportion=args.split_proportion,
            split_shuffle=args.split_shuffle,
            split_seed=args.seed,
            standardize=args.standardize,
            dtype=args.dtype,
            device=device,
        )
    return dataset


def set_precision(precision: torch.dtype):
    torch.set_default_dtype(precision)


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
    theta_unscaled: float,
    ntr: int,
    device: torch.device,
):
    # Get preconditioner config
    if preconditioner == "nystrom":
        precond_config = NystromConfig(
            rank=rank,
            rho=regularization,
            damping_mode=damping,
        )
    elif preconditioner == "identity":
        precond_config = IdentityConfig()
    elif preconditioner is None:
        precond_config = None
    else:
        raise ValueError(f"Unknown preconditioner: {preconditioner}")

    if opt_type == "pcg":
        max_iters = max_passes
    elif opt_type in ["sap", "sdd"]:
        max_iters = max_passes * blocks
    else:
        raise ValueError(f"Unknown optimization type: {opt_type}")

    # Get solver config
    solver_config_base_kwargs = {
        "device": device,
        "max_iters": max_iters,
        "atol": OPT_ATOL,
        "rtol": OPT_RTOL,
    }
    if opt_type == "pcg":
        solver_config = PCGConfig(
            precond_config=precond_config,
            **solver_config_base_kwargs,
        )
    elif opt_type == "sap":
        nu = float(blocks)
        mu = min(regularization, 1 / nu)  # Ensure mu * nu <= 1
        accel_config = SAPAccelConfig(mu=mu, nu=nu)
        solver_config = SAPConfig(
            precond_config=precond_config,
            blk_sz=ntr // blocks,
            accel_config=accel_config,
            **solver_config_base_kwargs,
        )
    elif opt_type == "sdd":
        solver_config = SDDConfig(
            momentum=OPT_SDD_MOMENTUM,
            step_size=step_size_unscaled / ntr,
            theta=theta_unscaled / max_iters,
            blk_size=ntr // blocks,
            **solver_config_base_kwargs,
        )

    return solver_config


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


def _get_saved_gp_hparams(
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


def get_gp_hparams(
    dataset_name: str,
    kernel_type: str,
    seed: int,
):
    if dataset_name in DOCKSTRING_DATASET_HPARAMS:
        return dockstring_hparams_to_gphparams(dataset_name)
    else:
        # Load the saved GP hyperparameters
        gp_hparams = _get_saved_gp_hparams(
            dataset_name,
            kernel_type,
            seed,
        )
        return gp_hparams


def get_rf_config(kernel_type: str, num_random_features: int):
    """
    Get the random features configuration based on the kernel type and number of
    random features.

    Args:
        kernel_type (str): The type of kernel used.
        num_random_features (int): The number of random features to use.

    Returns:
        RFConfig: The random features configuration.
    """
    if kernel_type in ["rbf", "matern12", "matern32", "matern52"]:
        return RFConfig(
            num_features=num_random_features,
        )
    else:
        raise ValueError(f"Unknown kernel type for random features: {kernel_type}")
