import subprocess
import argparse

from experiments.constants import (
    DATA_SPLIT_PROPORTION,
    DATA_SPLIT_SHUFFLE,
    DATA_STANDARDIZE,
    GP_TRAIN_DTYPE,
    GP_TRAIN_MAX_ITERS,
    GP_TRAIN_NUM_TRIALS,
    GP_TRAIN_SUBSAMPLE_SIZE,
    EXPERIMENT_DATA_KERNEL_MAP,
    EXPERIMENT_SEEDS,
)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run GP training with different datasets and seeds"
    )
    parser.add_argument("--device", type=str, help="Device ID")
    args = parser.parse_args()

    # Loop through datasets and seeds
    for dataset, kernel in EXPERIMENT_DATA_KERNEL_MAP.items():
        for seed in EXPERIMENT_SEEDS:
            cmd = [
                "python",
                "experiments/gp_training_base.py",
                "--dataset",
                dataset,
                "--kernel_type",
                kernel,
                "--seed",
                str(seed),
                "--device",
                str(args.device),
                "--split_proportion",
                str(DATA_SPLIT_PROPORTION),
                "--split_shuffle",
                str(DATA_SPLIT_SHUFFLE),
                "--standardize",
                str(DATA_STANDARDIZE),
                "--dtype",
                str(GP_TRAIN_DTYPE),
                "--subsample_size",
                str(GP_TRAIN_SUBSAMPLE_SIZE),
                "--num_trials",
                str(GP_TRAIN_NUM_TRIALS),
                "--max_iters",
                str(GP_TRAIN_MAX_ITERS),
            ]

            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd)


if __name__ == "__main__":
    main()
