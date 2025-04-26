import subprocess
import argparse

from experiments.constants import EXPERIMENT_DATA_KERNEL_MAP, EXPERIMENT_SEEDS


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
                "experiments/gp_training.py",
                "--dataset",
                dataset,
                "--kernel_type",
                kernel,
                "--seed",
                str(seed),
                "--device",
                str(args.device),
            ]

            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd)


if __name__ == "__main__":
    main()
