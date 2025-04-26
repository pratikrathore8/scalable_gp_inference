import argparse
import pickle
import os

from scalable_gp_inference.hparam_training import train_exact_gp_subsampled

from experiments.data_processing.load_torch import LOADERS
from experiments.constants import (
    DATA_NAMES,
    DATA_SPLIT_PROPORTION,
    DATA_SPLIT_SHUFFLE,
    DATA_STANDARDIZE,
    GP_TRAIN_DTYPE,
    GP_TRAIN_MAX_ITERS,
    GP_TRAIN_NUM_TRIALS,
    GP_TRAIN_OPT,
    GP_TRAIN_OPT_PARAMS,
    GP_TRAIN_SUBSAMPLE_SIZE,
    GP_TRAIN_SAVE_FILE_NAME,
)
from experiments.utils import device_type, set_random_seed, get_gp_hparams_save_file_dir


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
        choices=["rbf", "matern12", "matern32", "matern52"],
        help="The kernel type to use for the GP",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The random seed to use for training",
    )
    parser.add_argument(
        "--device",
        type=device_type,
        default="cpu",
        help="Device to use for training: 'cpu' or GPU device ID (non-neg. integer)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Set random seed for reproducibility
    set_random_seed(args.seed)

    # Load the dataset
    loader = LOADERS[args.dataset]
    dataset = loader(
        split_proportion=DATA_SPLIT_PROPORTION,
        split_shuffle=DATA_SPLIT_SHUFFLE,
        split_seed=args.seed,
        standardize=DATA_STANDARDIZE,
        dtype=GP_TRAIN_DTYPE,
        device=args.device,
    )

    # Train the GP model
    gp_hparams = train_exact_gp_subsampled(
        dataset.Xtr,
        dataset.ytr,
        kernel_type=args.kernel_type,
        opt_class=GP_TRAIN_OPT,
        opt_hparams=GP_TRAIN_OPT_PARAMS,
        training_iters=GP_TRAIN_MAX_ITERS,
        subsample_size=GP_TRAIN_SUBSAMPLE_SIZE,
        num_trials=GP_TRAIN_NUM_TRIALS,
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
