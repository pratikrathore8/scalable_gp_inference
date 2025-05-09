import argparse
import time

import torch
from rlaopt.kernels import KernelConfig
from scalable_gp_inference.bayes_opt.configs import BayesOptConfig, TSConfig
from scalable_gp_inference.bayes_opt.core import BayesOpt
import wandb

from experiments.constants import (
    BO_MAX_PASSES_PER_ITER,
    BO_MAX_ITERS,
    BO_NOISE_VARIANCE,
    BO_OPT_NUM_BLOCKS,
    BO_OPT_SDD_THETA_UNSCALED,
    BO_PRECISION,
    BO_KERNEL_TYPE,
    BO_KERNEL_CONST_SCALING,
    OPT_TYPES,
    OPT_RANK,
    OPT_DAMPING,
    OPT_SAP_PRECONDITIONERS,
    OPT_PCG_PRECONDITIONERS,
    OPT_SDD_STEP_SIZES_UNSCALED,
    LOGGING_USE_WANDB,
)
from experiments.utils import (
    device_type,
    get_solver_config,
    set_precision,
    set_random_seed,
)


OPT_PRECONDITIONERS_DICT = {
    "pcg": OPT_PCG_PRECONDITIONERS,
    "sap": OPT_SAP_PRECONDITIONERS,
}


def _get_bo_obj(
    bo_config: BayesOptConfig, device: torch.device, dtype: torch.dtype, seed: int
) -> BayesOpt:
    # Set the random seed each time to ensure the same set of initialization points
    # are used for each acquisition method and/or solver
    set_random_seed(seed)
    return BayesOpt(
        bo_config=bo_config,
        device=device,
        dtype=dtype,
    )


def _get_solver_config_kwargs_list(
    opt_types: str,
    max_passes: int,
    opt_preconditioners_dict: dict,
    rank: int,
    regularization: float,
    damping: str,
    blocks: int,
    step_sizes_unscaled: float,
    theta_unscaled: float,
    device: torch.device,
) -> list[dict]:
    # Loop over opt_types to get list of solver config kwargs (don't put in ntr though)
    solver_config_kwargs = []
    for opt_type in opt_types:
        if opt_type == "sdd":
            # Loop over the step sizes
            for step_size_unscaled in step_sizes_unscaled:
                solver_config_kwargs.append(
                    {
                        "opt_type": "sdd",
                        "max_passes": max_passes,
                        "preconditioner": None,
                        "rank": None,
                        "regularization": None,
                        "damping": None,
                        "blocks": blocks,
                        "step_size_unscaled": step_size_unscaled,
                        "theta_unscaled": theta_unscaled,
                        "device": device,
                    }
                )
        else:
            # Handle pcg and sap by looping over preconditioners
            for preconditioner in opt_preconditioners_dict[opt_type]:
                solver_config_kwargs.append(
                    {
                        "opt_type": opt_type,
                        "max_passes": max_passes,
                        "preconditioner": preconditioner,
                        "rank": rank,
                        "regularization": regularization,
                        "damping": damping,
                        "blocks": blocks,
                        "step_size_unscaled": None,
                        "theta_unscaled": None,
                        "device": device,
                    }
                )
    return solver_config_kwargs


def _run_single_experiment(
    bo_obj,
    ts_config: TSConfig,
    bo_config: BayesOptConfig,
    solver_config_kwargs: dict = None,
    wandb_config_dict: dict = {},
):
    """Run a single BO experiment with wandb logging."""
    # Configure wandb
    config_dict = {
        "max_passes_per_iter": solver_config_kwargs["max_passes"]
        if solver_config_kwargs
        else None,
        "bo_config": bo_config.to_dict(),
        "ts_config": ts_config.to_dict(),
        "solver_config": get_solver_config(
            ntr=len(bo_obj.bo_state), **solver_config_kwargs
        ).to_dict()
        if solver_config_kwargs
        else None,
        "solver_name": solver_config_kwargs["opt_type"]
        if solver_config_kwargs
        else None,
    }
    config_dict.update(wandb_config_dict)

    wandb.init(
        project=f"bayesopt_lengthscale_{bo_config.kernel_config.lengthscale}",
        config=config_dict,
    )

    # Log metrics at the start
    wandb.log(
        {
            "iter_time": 0.0,  # 0 since we haven't performed any optimization
            "num_acquisitions": len(bo_obj.bo_state),
            "fn_max": bo_obj.bo_state.fn_max,
            "fn_argmax": bo_obj.bo_state.fn_argmax,
        }
    )

    # Run experiment
    for _ in range(ts_config.num_iters):
        ts = time.time()
        krr_solver_config = None
        if solver_config_kwargs:
            krr_solver_config = get_solver_config(
                ntr=len(bo_obj.bo_state), **solver_config_kwargs
            )
        bo_obj.step(ts_config=ts_config, krr_solver_config=krr_solver_config)
        te = time.time()

        # Log metrics
        wandb.log(
            {
                "iter_time": te - ts,
                "num_acquisitions": len(bo_obj.bo_state),
                "fn_max": bo_obj.bo_state.fn_max,
                "fn_argmax": bo_obj.bo_state.fn_argmax,
            }
        )

    wandb.finish()


def _run_bo_experiment(
    bo_config: BayesOptConfig,
    device: torch.device,
    dtype: torch.dtype,
    bo_max_iters: int,
    use_wandb: bool,
    acquisition_method: str,
    seed: int,
    solver_config_kwargs_list: list[dict] = None,
    wandb_config_dict: dict = {},
):
    """
    Run Bayesian optimization experiment with the specified acquisition method.

    Args:
        bo_config: Bayesian optimization configuration
        device: PyTorch device
        dtype: PyTorch data type
        bo_max_iters: Maximum number of BO iterations
        use_wandb: Whether to use W&B logging
        acquisition_method: Acquisition method ("random_search" or "gp")
        solver_config_kwargs_list: List of solver configurations
          (only used for "nearby" method)
        wandb_config_dict: Additional W&B configuration
    """
    if not use_wandb:
        raise NotImplementedError(
            "Running experiment without wandb is not implemented yet. "
            "Please set LOGGING_USE_WANDB in constants.py to True "
            "to run the experiment."
        )

    ts_config = TSConfig(acquisition_method=acquisition_method, num_iters=bo_max_iters)

    # Start wandb config dict
    wandb_config_dict = {"precision": dtype, "seed": seed}

    # Handle different experiment methods
    if acquisition_method == "random_search":
        bo_obj = _get_bo_obj(bo_config=bo_config, device=device, dtype=dtype, seed=seed)
        _run_single_experiment(
            bo_obj=bo_obj,
            ts_config=ts_config,
            bo_config=bo_config,
            solver_config_kwargs=None,
            wandb_config_dict=wandb_config_dict,
        )
    elif acquisition_method == "gp":
        if not solver_config_kwargs_list:
            raise ValueError(
                "solver_config_kwargs_list must be provided for 'gp' method"
            )

        for solver_config_kwargs in solver_config_kwargs_list:
            try:
                bo_obj = _get_bo_obj(
                    bo_config=bo_config, device=device, dtype=dtype, seed=seed
                )
                _run_single_experiment(
                    bo_obj=bo_obj,
                    ts_config=ts_config,
                    bo_config=bo_config,
                    solver_config_kwargs=solver_config_kwargs,
                    wandb_config_dict=wandb_config_dict,
                )
            # Handle cases where the solver fails (e.g., SDD with large step size)
            except RuntimeError as e:
                print(f"Run failed for {solver_config_kwargs}")
                print(e)
    else:
        raise ValueError(f"Unknown acquisition method: {acquisition_method}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description=f"Run Bayesian optimization for a {BO_KERNEL_TYPE} kernel for a "
        "given lengthscale, seed, and device"
    )
    parser.add_argument(
        "--lengthscale",
        type=float,
        help=f"The lengthscale parameter for the {BO_KERNEL_TYPE} kernel",
    )
    parser.add_argument("--seed", type=int, help="The random seed to use")
    parser.add_argument("--device", type=device_type, help="Device ID")
    args = parser.parse_args()

    # Set precision for training
    set_precision(BO_PRECISION)

    # Set up parameters for kernel
    kernel_config = KernelConfig(
        const_scaling=BO_KERNEL_CONST_SCALING, lengthscale=args.lengthscale
    )

    # Get config for Bayesian optimization
    bo_config = BayesOptConfig(
        kernel_config=kernel_config,
        kernel_type=BO_KERNEL_TYPE,
        noise_variance=BO_NOISE_VARIANCE,
    )

    # Run Bayesian optimization with random search acquisition
    _run_bo_experiment(
        bo_config=bo_config,
        device=args.device,
        dtype=BO_PRECISION,
        bo_max_iters=BO_MAX_ITERS,
        use_wandb=LOGGING_USE_WANDB,
        acquisition_method="random_search",
        seed=args.seed,
        solver_config_kwargs_list=None,
    )

    # Run Bayesian optimization with GP-based acquisition
    # In this case, we actually use the PCG/SAP/SDD optimizers
    solver_config_kwargs_list = _get_solver_config_kwargs_list(
        opt_types=OPT_TYPES,
        max_passes=BO_MAX_PASSES_PER_ITER,
        opt_preconditioners_dict=OPT_PRECONDITIONERS_DICT,
        rank=OPT_RANK,
        regularization=BO_NOISE_VARIANCE,
        damping=OPT_DAMPING,
        blocks=BO_OPT_NUM_BLOCKS,
        step_sizes_unscaled=OPT_SDD_STEP_SIZES_UNSCALED,
        theta_unscaled=BO_OPT_SDD_THETA_UNSCALED,
        device=args.device,
    )
    _run_bo_experiment(
        bo_config=bo_config,
        device=args.device,
        dtype=BO_PRECISION,
        bo_max_iters=BO_MAX_ITERS,
        use_wandb=LOGGING_USE_WANDB,
        acquisition_method="gp",
        seed=args.seed,
        solver_config_kwargs_list=solver_config_kwargs_list,
    )


if __name__ == "__main__":
    main()
