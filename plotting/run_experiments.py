import argparse
import copy
import datetime as _dt
import importlib
import pickle
from pathlib import Path
from typing import Any, Mapping

import torch

# -----------------------------------------------------------------------------
# Configuration ----------------------------------------------------------------
# -----------------------------------------------------------------------------
try:
    from scalable_gp_inference.plotting.experiments_config import runs
except ImportError:
    from experiments_config import runs

from experiments.data_processing.load_torch import LOADERS
from scalable_gp_inference.gp_inference import GPInference
from scalable_gp_inference.hparam_training import GPHparams

_RESULTS_DIR = Path(".results")
_RESULTS_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Utility: instantiate recursively --------------------------------------------
# -----------------------------------------------------------------------------

def _instantiate(
        cfg: Mapping[str, Any], *, extra_kwargs: dict[str, Any] | None = None
):
    """Recursively build an object from a ``{"class": "pkg.module.Class", "kwargs": {...}}`` descriptor."""

    cls_path = cfg["class"]
    kwargs = copy.deepcopy(cfg.get("kwargs", {}))

    # Recurse on nested descriptors (e.g. preconditioner configs)
    for k, v in kwargs.items():
        if isinstance(v, dict) and "class" in v:
            kwargs[k] = _instantiate(v)

    if extra_kwargs:
        kwargs.update(extra_kwargs)

    module_name, _, attr_name = cls_path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)(**kwargs)

# -----------------------------------------------------------------------------
# Core helpers -----------------------------------------------------------------
# -----------------------------------------------------------------------------

def save_progress(result: dict, meta: dict[str, Any]):
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{meta['dataset']}_{meta['solver']}_{stamp}.pkl"
    with open(_RESULTS_DIR / fname, "wb") as fh:
        pickle.dump({"meta": meta, "result": result}, fh)
    print(f"[✓] Saved results to {_RESULTS_DIR / fname}")


def run_experiment(dataset: str, solver: str, *, eval_freq: int = 10):
    """Run *one* dataset/solver pair as described in `experiments_config`."""

    # ---------------------------------------------------------------------
    # 0.  Gather configuration snippets -----------------------------------
    # ---------------------------------------------------------------------
    dataset_entry = runs["datasets"][dataset]
    dataset_configs = dataset_entry["dataset_configs"]
    gp_hyperparams_configs = dataset_entry["gp_hyperparams"]
    use_full_kernel = gp_hyperparams_configs["use_full_kernel"]
    solver_configs = runs["solvers"][solver]

    # ---------------------------------------------------------------------
    # 1.  Load data --------------------------------------------------------
    # ---------------------------------------------------------------------
    dtype = getattr(torch, dataset_configs["dtype"])
    device = torch.device(dataset_configs["device"])

    loader_fn = LOADERS[dataset_configs["loader"]]
    split = loader_fn(
        split_proportion = dataset_configs["split_proportion"],
        split_shuffle = dataset_configs["split_shuffle"],
        split_seed = dataset_configs["split_seed"],
        standardize = dataset_configs["standardize"],
        dtype = dtype,
        device = device,
    )

    Xtr, Xtst, ytr, ytst = split.Xtr, split.Xtst, split.ytr, split.ytst


    # ---------------------------------------------------------------------
    # 2.  GP hyper‑parameters ---------------------------------------------
    # ---------------------------------------------------------------------
    hp = GPHparams(
        noise_variance = gp_hyperparams_configs["noise_variance"],
        signal_variance = gp_hyperparams_configs["signal_variance"],
        kernel_lengthscale = torch.as_tensor(gp_hyperparams_configs["kernel_lengthscale"], dtype=dtype, device=device),
    )


    gp = GPInference(
        Xtr, ytr, Xtst, ytst,
        kernel_type = gp_hyperparams_configs["kernel_type"],
        kernel_hparams = hp,
        num_posterior_samples = 0,
    )

    # ---------------------------------------------------------------------
    # 3.  Solver configuration --------------------------------------------
    # ---------------------------------------------------------------------
    # Inject device into top‑level solver kwargs --------------------------
    solver_configs = copy.deepcopy(solver_configs)
    solver_configs.setdefault("kwargs", {})["device"] = device

    # Patch rho in preconditioner if asked ("rho": "auto")
    pre = solver_configs["kwargs"].get("precond_config")
    if pre and isinstance(pre, dict):
        if isinstance(pre["kwargs"].get("rho"), str) and pre["kwargs"]["rho"].lower() == "auto":
            pre["kwargs"]["rho"] = gp_hyperparams_configs["noise_variance"]

    solver_instance = _instantiate(solver_configs)

    # Auto blk_sz for SAP if missing
    if solver_configs["class"].endswith("SAPConfig") and not getattr(solver_instance, "blk_sz", None):
        solver_instance.blk_sz = max(1, Xtr.shape[0] // 100)


    # ---------------------------------------------------------------------
    # 4.  Inference --------------------------------------------------------
    # ---------------------------------------------------------------------
    result = gp.perform_inference(
        solver_config = solver_instance,
        W_init = None,
        use_full_kernel = use_full_kernel,
        eval_freq = eval_freq,
        log_in_wandb = False,
        wandb_init_kwargs = {}

    )

    # ---------------------------------------------------------------------
    # 5.  Persist ---------------------------------------------------------
    # ---------------------------------------------------------------------
    meta = {
        "dataset": dataset,
        "solver": solver,
        "dataset_configs": dataset_configs,
        "gp_hyperparams_configs": gp_hyperparams_configs,
        "solver_configs": solver_configs,
    }
    save_progress(result, meta)

    return result


def run_all_experiments():
    for ds_name in runs["datasets"]:
        for sol_name in runs["solvers"]:
            print(f"\n=== Running {ds_name} × {sol_name} ===")
            run_experiment(ds_name, sol_name)

# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scalable-GP experiments.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset key (see experiments_config.py)")
    parser.add_argument("--solver",  type=str, default=None, help="Solver key (see experiments_config.py)")
    parser.add_argument("--eval_freq", type=int, default=10, help="Callback / logging frequency")
    args = parser.parse_args()

    if args.dataset and args.solver:
        run_experiment(args.dataset, args.solver, eval_freq=args.eval_freq)
    else:
        run_all_experiments()
