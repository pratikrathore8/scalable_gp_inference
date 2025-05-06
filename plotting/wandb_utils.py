from typing import Iterable, Optional, Dict, List
import pandas as pd
import wandb
from wandb_constants import CONFIG_KEYS, METRIC_PATHS


def get_project_runs(
        entity: str,
        project: str,
        filters: Optional[Dict] = None,
        max_runs: Optional[int] = None
) -> List[wandb.apis.public.Run]:
    """Fetch runs with optional filters and limit."""
    api = wandb.Api()
    try:
        runs = api.runs(f"{entity}/{project}", order="-created_at",
                        filters=filters)
        if max_runs:
            print(
                f"Total number of runs found: {len(runs)} | Keeping {max_runs} runs")
        else:
            print(f"Total number of runs found: {len(runs)}")
        return runs[:max_runs] if max_runs else runs
    except wandb.CommError as e:
        print(f"API Error: {str(e)}")
        return []


def get_run_history(
        run: wandb.apis.public.Run,
        metrics: List[str],
        x_axis: str = "step",
        include_config: bool = True
) -> pd.DataFrame:
    """Get history using pre-fetched run object"""
    try:
        required_keys = list(set(metrics + ["_step", "cum_time", "iter_time"]))
        history = run.scan_history(keys=required_keys)
        df = pd.DataFrame(history).astype(float, errors="ignore")
        df.sort_values("_step", inplace=True)

        if x_axis == "time":
            df["x_value"] = df["cum_time"]

        elif x_axis == "datapasses":
            config = run.config
            solver = config.get(CONFIG_KEYS["SOLVER"], "").lower()

            if solver in ["sap", "sdd"]:
                opt_num_blocks = config.get("opt_num_blocks", 1)
                df["x_value"] = df["_step"] / opt_num_blocks
            elif solver == "pcg":
                df["x_value"] = df["_step"]
            else:
                n = config.get("ntr", 1)
                m = config.get("rf_config", {}).get("num_features", 0)
                eval_freq = config.get("eval_freq", 1)
                scaling = (2 * m * eval_freq) / n if m else eval_freq / n
                df["x_value"] = df["_step"] * scaling

        else:
            df["x_value"] = df["_step"]

        if include_config:
            for k, v in run.config.items():
                if isinstance(v, (int, float, str, bool)):
                    df[k] = v

        return df[["x_value"] + metrics]
    except Exception as e:
        print(f"Error processing run {run.id}: {str(e)}")
        return pd.DataFrame()


def organize_runs_data(
        runs: Iterable[wandb.apis.public.Run],
        y_metrics: List[str],
        x_axis: str
) -> Dict[str, pd.DataFrame]:
    data = {}
    for run in runs:
        try:
            df = get_run_history(run, y_metrics, x_axis=x_axis)
            if not df.empty:
                df["run_id"] = run.id
                df["run_name"] = run.name
                solver = run.config.get(CONFIG_KEYS["SOLVER"], "unknown")
                df[CONFIG_KEYS["SOLVER"]] = solver
                if solver == "sap":
                    solver_config = run.config.get("solver_config", {})
                    df[
                        "sap_precond"] = "Nystrom" if "precond_config" in solver_config else "Identity"
                df[CONFIG_KEYS["DATASET"]] = run.config.get(
                    CONFIG_KEYS["DATASET"], "unknown")
                data[run.id] = df
        except Exception as e:
            print(f"Critical error with run {run.id}: {str(e)}")
    return data


def filter_runs(
        runs: Iterable[wandb.apis.public.Run],
        require_all: Optional[Dict] = None,
        require_any: Optional[Dict] = None
) -> List[wandb.apis.public.Run]:
    """Flexible run filtering with list support"""
    filtered = []
    for run in runs:
        config = run.config
        match_all = True

        if require_all:
            for k, v in require_all.items():
                config_val = config.get(k)
                if isinstance(v, list):
                    if config_val not in v:
                        match_all = False
                        break
                else:
                    if config_val != v:
                        match_all = False
                        break

        match_any = False
        if require_any:
            for k, v in require_any.items():
                config_val = config.get(k)
                if isinstance(v, list):
                    if config_val in v:
                        match_any = True
                        break
                else:
                    if config_val == v:
                        match_any = True
                        break

        if (not require_all or match_all) and (not require_any or match_any):
            filtered.append(run)

    print(f"Number of runs after filtering: {len(filtered)}")
    return filtered


def choose_runs(
        runs: List[wandb.apis.public.Run],
        strategy_map: Dict[str, str], # example: {"sap": "best", "pcg": "latest"}
        metric: str,
        metric_agg: str
) -> List[wandb.apis.public.Run]:
    """
    Select runs using per-solver strategies while preserving unspecified solvers.

    Args:
        strategy_map: Dictionary mapping solver names to strategies ("best"/"latest")
        metric: Metric path for "best" strategy evaluation
        metric_agg: Aggregation method ("last"/"min"/"max")
    """
    from collections import defaultdict

    solver_groups = defaultdict(list)
    for run in runs:
        if solver := run.config.get(CONFIG_KEYS["SOLVER"]):
            if solver == "sap":
                solver_config = run.config.get("solver_config", {})
                precond_type = "Nystrom" if "precond_config" in solver_config else "Identity"
                solver_groups[(solver, precond_type)].append(run)
            else:
                solver_groups[solver].append(run)

    selected = []

    for solver, s_runs in solver_groups.items():
        if isinstance(solver, tuple):
            solver_name, precond_type = solver
            strategy = strategy_map.get(solver_name, None)
            label = f"{solver_name} ({precond_type})"
            print(
                f"Applying '{strategy}' selection for {label} ({len(s_runs)} runs)")
        else:
            strategy = strategy_map.get(solver, None)
            print(
                f"Applying '{strategy}' selection for {solver} ({len(s_runs)} runs)")

        if strategy == "latest":
            selected.append(max(s_runs, key=lambda r: r.created_at))

        elif strategy == "best":
            metrics = []
            for run in s_runs:
                try:
                    df = get_run_history(run, [metric])
                    if df.empty:
                        print(f"Skipping run {run.id} - empty dataframe")
                        continue

                    if metric_agg == "last":
                        val = df[metric].iloc[-1]
                    elif metric_agg == "min":
                        val = df[metric].min()
                    elif metric_agg == "max":
                        val = df[metric].max()
                    else:
                        val = df[metric].iloc[-1]

                    if not np.isfinite(val):
                        print(
                            f"Skipping run {run.id} - invalid metric value: {val}")
                        continue

                    metrics.append((val, run))
                    # print(f"Run {run.id[:8]} metric {metric}: {val} (agg: {metric_agg})")
                except Exception as e:
                    print(f"Skipping run {run.id}: {str(e)}")
                    continue

            if metrics:
                if metric in {METRIC_PATHS["TEST_RMSE"],
                              METRIC_PATHS["POSTERIOR_NLL"],
                              METRIC_PATHS["POSTERIOR_MEAN_NLL"]}:
                    best_run = min(metrics, key=lambda x: x[0])[1]
                else:
                    best_run = max(metrics, key=lambda x: x[0])[1]

                # print(f"Selected best run {best_run.id[:8]} with metric value: {min(metrics, key=lambda x: x[0])[0]}")
                selected.append(best_run)

    return selected