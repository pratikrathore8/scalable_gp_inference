import argparse
import pickle
import os

from scalable_gp_inference.hparam_training import train_exact_gp_subsampled

from experiments.constants import (
    DATA_NAMES,
    EXPERIMENT_KERNELS,
    GP_TRAIN_OPT,
    GP_TRAIN_OPT_PARAMS,
    GP_TRAIN_SAVE_FILE_NAME,
)
from experiments.utils import (
    device_type,
    dtype_type,
    load_dataset,
    set_random_seed,
    set_precision,
    get_gp_hparams_save_file_dir,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a GP on a dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATA_NAMES,
        help="The name of the dataset to train on",
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
        help="The random seed to use for training",
    )
    parser.add_argument(
        "--device",
        type=device_type,
        help="Device to use for training: 'cpu' or GPU device ID (non-neg. integer)",
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
        help="Data type to use for training (e.g., 'float32', 'float64')",
    )
    parser.add_argument(
        "--subsample_size",
        type=int,
        help="Subsample size for training",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        help="Number of trials for training",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        help="Maximum number of iterations for training",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Set random seed for reproducibility
    set_random_seed(args.seed)

    # Set precision for training
    set_precision(args.dtype)

    # Load the dataset
    dataset = load_dataset(args, device=args.device)

    # Train the GP model
    gp_hparams = train_exact_gp_subsampled(
        dataset.Xtr,
        dataset.ytr,
        kernel_type=args.kernel_type,
        opt_class=GP_TRAIN_OPT,
        opt_hparams=GP_TRAIN_OPT_PARAMS,
        training_iters=args.max_iters,
        subsample_size=args.subsample_size,
        num_trials=args.num_trials,
    )

    # Save the trained GP hyperparameters
    save_file_dir = get_gp_hparams_save_file_dir(
        dataset_name=args.dataset,
        kernel_type=args.kernel_type,
        seed=args.seed,
    )
    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)
    save_file_path = os.path.join(save_file_dir, GP_TRAIN_SAVE_FILE_NAME)
    with open(save_file_path, "wb") as f:
        pickle.dump(gp_hparams.to("cpu"), f)


if __name__ == "__main__":
    main()
