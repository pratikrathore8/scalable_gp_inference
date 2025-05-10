import subprocess
import argparse

from experiments.constants import (
    DATA_SPLIT_PROPORTION_MAP,
    DATA_SPLIT_SHUFFLE,
    DATA_STANDARDIZE,
    EXPERIMENT_DATA_KERNEL_MAP,
    GP_INFERENCE_NUM_POSTERIOR_SAMPLES_MAP,
    GP_INFERENCE_NUM_RANDOM_FEATURES,
    GP_INFERENCE_USE_FULL_KERNEL_MAP,
    OPT_TYPES,
    OPT_RANK,
    OPT_DAMPING,
    OPT_SAP_PRECONDITIONERS,
    OPT_SAP_PRECISIONS,
    OPT_PCG_PRECONDITIONERS,
    OPT_PCG_PRECISIONS,
    OPT_SDD_STEP_SIZES_UNSCALED,
    OPT_SDD_THETA_UNSCALED,
    OPT_SDD_PRECISIONS,
    OPT_MAX_PASSES_MAP,
    OPT_MAX_PASSES_TIMING,
    OPT_NUM_BLOCKS_MAP,
    LOGGING_USE_WANDB,
    LOGGING_EVAL_FREQ_MAP,
    LOGGING_EVAL_FREQ_MAP_TAXI,
)


def _get_precision_extensions(precisions):
    extension_list = []
    for precision in precisions:
        if precision in ["float32", "float64"]:
            extension_list.append(
                [
                    "--dtype",
                    precision,
                ]
            )
        else:
            raise ValueError(f"Unknown precision: {precision}")
    return extension_list


def _get_precond_extensions(preconditioners):
    extension_list = []
    for precond in preconditioners:
        if precond == "nystrom":
            extension_list.append(
                [
                    "--opt_preconditioner",
                    precond,
                    "--opt_rank",
                    str(OPT_RANK),
                    "--opt_damping",
                    OPT_DAMPING,
                ]
            )
        elif precond == "identity":
            extension_list.append(
                [
                    "--opt_preconditioner",
                    precond,
                ]
            )
        else:
            raise ValueError(f"Unknown preconditioner: {precond}")
    return extension_list


def _get_pcg_extensions(eval_freq_map):
    base_extension = [
        "--eval_freq",
        str(eval_freq_map["pcg"]),
        "--opt_type",
        "pcg",
    ]
    precond_extensions = _get_precond_extensions(OPT_PCG_PRECONDITIONERS)
    precision_extensions = _get_precision_extensions(OPT_PCG_PRECISIONS)
    # Go through each precision and preconditioner extension
    # and add the base extension to the start of each combination
    extensions = [
        [*base_extension, *precond, *precision]
        for precond in precond_extensions
        for precision in precision_extensions
    ]
    return extensions


def _get_sap_extensions(dataset, eval_freq_map):
    base_extension = [
        "--eval_freq",
        str(eval_freq_map["sap"]),
        "--opt_type",
        "sap",
        "--opt_num_blocks",
        str(OPT_NUM_BLOCKS_MAP[dataset]),
    ]
    precond_extensions = _get_precond_extensions(OPT_SAP_PRECONDITIONERS)
    precision_extensions = _get_precision_extensions(OPT_SAP_PRECISIONS)
    # Go through each precision and preconditioner extension
    # and add the base extension to the start of each precision extension
    extensions = [
        [*base_extension, *precond, *precision]
        for precond in precond_extensions
        for precision in precision_extensions
    ]
    return extensions


def _get_sdd_extensions(dataset, eval_freq_map):
    base_extension = [
        "--eval_freq",
        str(eval_freq_map["sdd"]),
        "--opt_type",
        "sdd",
        "--opt_num_blocks",
        str(OPT_NUM_BLOCKS_MAP[dataset]),
        "--theta_unscaled",
        str(OPT_SDD_THETA_UNSCALED),
    ]
    precision_extensions = _get_precision_extensions(OPT_SDD_PRECISIONS)
    # Go through each precision and unscaled step size and add the base extension
    # to the start of each combination
    extensions = [
        [*base_extension, *precision, *["--opt_step_size_unscaled", str(step_size)]]
        for step_size in OPT_SDD_STEP_SIZES_UNSCALED
        for precision in precision_extensions
    ]
    return extensions


def _get_base_command(args):
    cmd = [
        "python",
        "experiments/gp_inference_base.py",
        "--dataset",
        args.dataset,
        "--kernel_type",
        EXPERIMENT_DATA_KERNEL_MAP[args.dataset],
        "--seed",
        str(args.seed),
        "--devices",
    ]

    # Add each device as a separate item
    cmd.extend(args.devices)

    cmd.extend(
        [
            "--split_proportion",
            str(DATA_SPLIT_PROPORTION_MAP[args.dataset]),
        ]
    )

    # Only add flags if conditions are True
    if DATA_SPLIT_SHUFFLE:
        cmd.append("--split_shuffle")
    if DATA_STANDARDIZE:
        cmd.append("--standardize")

    cmd.extend(
        [
            "--num_posterior_samples",
            str(GP_INFERENCE_NUM_POSTERIOR_SAMPLES_MAP[args.dataset]),
            "--num_random_features",
            str(GP_INFERENCE_NUM_RANDOM_FEATURES),
        ]
    )

    if GP_INFERENCE_USE_FULL_KERNEL_MAP[args.dataset]:
        cmd.append("--use_full_kernel")
    if LOGGING_USE_WANDB:
        cmd.append("--log_in_wandb")

    cmd.extend(
        [
            "--opt_max_passes",
            str(OPT_MAX_PASSES_MAP[args.dataset])
            if not args.timing
            else str(OPT_MAX_PASSES_TIMING),
        ]
    )
    if args.timing:
        cmd.append("--timing")

    return cmd


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run GP inference with a given dataset and seed"
    )
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--seed", type=int, help="The random seed to use")
    parser.add_argument("--devices", type=str, nargs="+", help="Device IDs")
    parser.add_argument(
        "--timing", action="store_true", help="Whether to run timing experiments"
    )
    args = parser.parse_args()

    # Get the base command
    base_cmd = _get_base_command(args)

    # Get the appropriate map for logging eval frequency
    if args.dataset == "taxi":
        eval_freq_map = LOGGING_EVAL_FREQ_MAP_TAXI
    else:
        eval_freq_map = LOGGING_EVAL_FREQ_MAP

    # Get the extensions for all the optimizers
    opt_extensions = {}
    for opt_type in OPT_TYPES:
        if opt_type == "pcg":
            opt_extensions[opt_type] = _get_pcg_extensions(eval_freq_map)
        elif opt_type == "sap":
            opt_extensions[opt_type] = _get_sap_extensions(args.dataset, eval_freq_map)
        elif opt_type == "sdd":
            opt_extensions[opt_type] = _get_sdd_extensions(args.dataset, eval_freq_map)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

    # Loop through each optimizer and its extensions and run GP inference
    for opt_type, extensions in opt_extensions.items():
        for extension in extensions:
            # Combine the base command with the optimizer-specific extensions
            cmd = base_cmd + extension
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd)


if __name__ == "__main__":
    main()
