import argparse

from scalable_gp_inference.gp_inference import GPInference

from experiments.constants import (
    DATA_NAMES,
    EXPERIMENT_KERNELS,
    LOGGING_WANDB_PROJECT_BASE_NAME,
    OPT_TYPES,
)
from experiments.utils import (
    device_type,
    dtype_type,
    none_or_str,
    load_dataset,
    set_random_seed,
    set_precision,
    get_solver_config,
    get_gp_hparams,
    get_rf_config,
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
        action="store_true",
        help="Whether to shuffle the data before splitting",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
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
        action="store_true",
        help="Whether to use the full kernel during inference",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        help="Frequency of evaluation during inference",
    )
    parser.add_argument(
        "--log_in_wandb",
        action="store_true",
        help="Whether to log results in Weights & Biases",
    )
    parser.add_argument(
        "--opt_type",
        type=str,
        choices=OPT_TYPES,
        help="Type of optimization algorithm to use",
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
        "--opt_num_blocks",
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
    parser.add_argument(
        "--theta_unscaled",
        type=float,
        default=None,
        help="Averaging parameter for the optimizer -- SDD only",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Set random seed for reproducibility
    set_random_seed(args.seed)

    # Set precision for training
    set_precision(args.dtype)

    # Load the GP hyperparameters
    gp_hparams = get_gp_hparams(args.dataset, args.kernel_type, args.seed)
    gp_hparams = gp_hparams.to(device=args.devices[0], dtype=args.dtype)

    # Load the dataset for inference
    dataset = load_dataset(args, device=args.devices[0])

    # Get solver configuration
    solver_config = get_solver_config(
        opt_type=args.opt_type,
        max_passes=args.opt_max_passes,
        preconditioner=args.opt_preconditioner,
        rank=args.opt_rank,
        regularization=gp_hparams.noise_variance,
        damping=args.opt_damping,
        blocks=args.opt_num_blocks,
        step_size_unscaled=args.opt_step_size_unscaled,
        theta_unscaled=args.theta_unscaled,
        ntr=dataset.Xtr.shape[0],
        device=args.devices[0],
    )

    # Get random features configuration
    rf_config = get_rf_config(
        kernel_type=args.kernel_type, num_random_features=args.num_random_features
    )

    # Get GP inference object
    model = GPInference(
        Xtr=dataset.Xtr,
        ytr=dataset.ytr,
        Xtst=dataset.Xtst,
        ytst=dataset.ytst,
        kernel_type=args.kernel_type,
        kernel_hparams=gp_hparams,
        num_posterior_samples=args.num_posterior_samples,
        rf_config=rf_config,
        distributed=len(args.devices) > 1,
        devices=set(args.devices),
    )

    wandb_init_kwargs = {
        "project": f"{LOGGING_WANDB_PROJECT_BASE_NAME}_{args.dataset}",
        "config": {
            "dataset": args.dataset,
            "ntr": dataset.Xtr.shape[0],
            "ntst": dataset.Xtst.shape[0],
            "p": dataset.Xtr.shape[1],
            "kernel_type": args.kernel_type,
            "gp_hparams": gp_hparams.to_dict(),
            "num_posterior_samples": args.num_posterior_samples,
            "rf_config": rf_config,
            "seed": args.seed,
            "all_devices": args.devices,
            "max_passes": args.opt_max_passes,
            "opt_num_blocks": args.opt_num_blocks,
            "opt_step_size_unscaled": args.opt_step_size_unscaled,
            "theta_unscaled": args.theta_unscaled,
            "use_full_kernel": args.use_full_kernel,
            "eval_freq": args.eval_freq,
        },
    }

    # Run inference
    results = model.perform_inference(
        solver_config=solver_config,
        W_init=None,
        use_full_kernel=args.use_full_kernel,
        eval_freq=args.eval_freq,
        log_in_wandb=args.log_in_wandb,
        wandb_init_kwargs=wandb_init_kwargs if args.log_in_wandb else {},
    )

    return results


if __name__ == "__main__":
    main()
