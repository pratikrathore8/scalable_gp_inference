from typing import Iterable, Optional, Dict, List
import pandas as pd
import numpy as np
from collections import defaultdict
import wandb
import warnings
from wandb_constants import (
    CONFIG_KEYS,
    METRIC_PATHS,
    BAYESIAN_OPT_CONFIG_KEYS,
    BAYESIAN_OPT_METRIC_PATHS,
)


def get_project_runs(
    entity: str,
    project: str,
    filters: Optional[Dict] = None,
    max_runs: Optional[int] = None,
) -> List[wandb.apis.public.Run]:
    """Fetch runs with optional filters and limit."""
    api = wandb.Api()
    try:
        runs = api.runs(f"{entity}/{project}", order="-created_at", filters=filters)
        if max_runs:
            print(
                f"Total number of runs found for {project}: {len(runs)} | Keeping {max_runs} runs"
            )
        else:
            print(f"Total number of runs found for {project}: {len(runs)}")
        return runs[:max_runs] if max_runs else runs
    except wandb.CommError as e:
        print(f"API Error: {str(e)}")
        return []


def get_run_history(
    is_bayesian_opt: bool,
    run: wandb.apis.public.Run,
    metrics: List[str],
    x_axis: str = "step",
    include_config: bool = True,
) -> pd.DataFrame:
    """Get history using pre-fetched run object"""
    if is_bayesian_opt:
        try:
            required_keys = list(
                set(metrics + ["_step", "iter_time", "num_acquisitions"])
            )
            history = run.scan_history(keys=required_keys)
            df = pd.DataFrame(history).astype(float, errors="ignore")
            if "_step" not in df.columns:
                raise KeyError("_step")
            df.sort_values("_step", inplace=True)

            if x_axis == "time":
                df["x_value"] = df["iter_time"].cumsum()

            elif x_axis == "num_acquisitions":
                df["x_value"] = df["num_acquisitions"]

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

    else:
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
    is_bayesian_opt: bool,
    runs: Iterable[wandb.apis.public.Run],
    y_metrics: List[str],
    x_axis: str,
) -> Dict[str, pd.DataFrame]:
    data = {}
    for run in runs:
        try:
            df = get_run_history(is_bayesian_opt, run, y_metrics, x_axis=x_axis)
            if not df.empty:
                df["run_id"] = run.id
                df["run_name"] = run.name
                if is_bayesian_opt:
                    solver = run.config.get(
                        BAYESIAN_OPT_CONFIG_KEYS["SOLVER"], "unknown"
                    )
                    df[BAYESIAN_OPT_CONFIG_KEYS["SOLVER"]] = solver
                else:
                    df[CONFIG_KEYS["DATASET"]] = run.config.get(
                        CONFIG_KEYS["DATASET"], "unknown"
                    )
                    solver = run.config.get(CONFIG_KEYS["SOLVER"], "unknown")
                    df[CONFIG_KEYS["SOLVER"]] = solver

                if solver == "sap":
                    solver_config = run.config.get("solver_config", {})
                    df["sap_precond"] = (
                        "Nystrom" if "precond_config" in solver_config else "Identity"
                    )
                data[run.id] = df
        except Exception as e:
            print(f"Critical error with run {run.id}: {str(e)}")
    return data


def filter_runs(
    runs: Iterable[wandb.apis.public.Run],
    require_all: Optional[Dict] = None,
    require_any: Optional[Dict] = None,
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
    is_bayesian_opt: bool,
    runs: List[wandb.apis.public.Run],
    strategy_map: Dict[str, str],
    metric: str,
    metric_agg: str,
) -> List[wandb.apis.public.Run]:
    """
    Select runs using per-solver strategies while preserving unspecified solvers.

    Args:
        strategy_map: Dictionary mapping solver names to strategies ("best"/"latest")
        metric: Metric path for "best" strategy evaluation
        metric_agg: Aggregation method ("last"/"min"/"max")
    """

    solver_groups = defaultdict(list)
    for run in runs:
        solver = run.config.get(
            CONFIG_KEYS["SOLVER"]
            if not is_bayesian_opt
            else BAYESIAN_OPT_CONFIG_KEYS["SOLVER"]
        )
        if solver:
            if solver == "sap":
                solver_config = run.config.get("solver_config", {})
                precond_type = (
                    "Nystrom" if "precond_config" in solver_config else "Identity"
                )
                solver_groups[(solver, precond_type)].append(run)
            else:
                solver_groups[solver].append(run)

    selected = []

    for solver, s_runs in solver_groups.items():
        if isinstance(solver, tuple):
            solver_name, precond_type = solver
            strategy = strategy_map.get(solver_name, None)
            label = f"{solver_name} ({precond_type})"
            print(f"Applying '{strategy}' selection for {label} ({len(s_runs)} runs)")
        else:
            strategy = strategy_map.get(solver, None)
            print(f"Applying '{strategy}' selection for {solver} ({len(s_runs)} runs)")

        if strategy == "latest":
            selected.append(max(s_runs, key=lambda r: r.created_at))

        elif strategy == "best":
            metrics = []
            for run in s_runs:
                try:
                    df = get_run_history(is_bayesian_opt, run, [metric])
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
                        print(f"Skipping run {run.id} - invalid metric value: {val}")
                        continue

                    metrics.append((val, run))
                    # print(f"Run {run.id[:8]} metric {metric}: {val} (agg: {metric_agg})")
                except Exception as e:
                    print(f"Skipping run {run.id}: {str(e)}")
                    continue

            if metrics:
                if metric in {
                    METRIC_PATHS["TEST_RMSE"],
                    METRIC_PATHS["POSTERIOR_NLL"],
                    METRIC_PATHS["POSTERIOR_MEAN_NLL"],
                }:
                    best_run = min(metrics, key=lambda x: x[0])[1]
                else:
                    best_run = max(metrics, key=lambda x: x[0])[1]

                # print(f"Selected best run {best_run.id[:8]} with metric value: {min(metrics, key=lambda x: x[0])[0]}")
                selected.append(best_run)

    return selected


def choose_and_aggregate_runs(
    is_bayesian_opt: bool,
    runs: List[wandb.apis.public.Run],
    y_metrics: List[str],
    num_seeds: int,
    sort_metric: str,
) -> Dict[str, pd.DataFrame]:
    """
    For plotting errorbars only. Takes the top (num_seeds) number of filtered runs, and for each metric, extracts the mean and std.

    Returns {solver_label: DataFrame} where each DataFrame holds one row with the final value, mean and std for every metric in y_metrics.
    """
    solver_groups = defaultdict(list)
    for run in runs:
        solver = run.config.get(
            CONFIG_KEYS["SOLVER"]
            if not is_bayesian_opt
            else BAYESIAN_OPT_CONFIG_KEYS["SOLVER"]
        )
        if solver:
            if solver == "sap":
                solver_config = run.config.get("solver_config", {})
                precond_type = (
                    "Nystrom" if "precond_config" in solver_config else "Identity"
                )
                solver_groups[(solver, precond_type)].append(run)
            else:
                solver_groups[solver].append(run)

    grouped_runs = {}
    for solver, s_runs in solver_groups.items():
        label = (
            f"{solver[0]} ({solver[1]})" if isinstance(solver, tuple) else str(solver)
        )
        grouped_runs[label] = s_runs

    aggregated: dict[str, pd.DataFrame] = {}
    for label, runs in grouped_runs.items():
        records = {m: [] for m in y_metrics}

        if num_seeds is not None and num_seeds < len(runs):
            use_metric = sort_metric or y_metrics[0]
            lower_is_better = None

            if not is_bayesian_opt:
                lower_is_better = use_metric in {
                    METRIC_PATHS["TEST_RMSE"],
                    METRIC_PATHS["POSTERIOR_NLL"],
                    METRIC_PATHS["POSTERIOR_MEAN_NLL"],
                }

            scored: list[tuple[float, wandb.apis.public.Run]] = []
            for r in runs:
                df = get_run_history(
                    is_bayesian_opt,
                    r,
                    [use_metric],
                    x_axis="datapasses",
                    include_config=False,
                )
                if df.empty or not np.isfinite(df[use_metric].iloc[-1]):
                    continue
                scored.append((df[use_metric].iloc[-1], r))

            scored.sort(key=lambda t: t[0], reverse=not lower_is_better)
            runs = [r for _, r in scored[:num_seeds]]

        records = {m: [] for m in y_metrics}

        for run in runs:
            df = get_run_history(
                is_bayesian_opt,
                run,
                y_metrics,
                x_axis="datapasses",
                include_config=False,
            )
            if df.empty:
                continue
            # take the final value
            for m in y_metrics:
                records[m].append(df[m].iloc[-1])

        if not all(records[m] for m in y_metrics):  # skip empty sets
            continue

        stats = {f"{m}_mean": np.mean(records[m]) for m in y_metrics} | {
            f"{m}_std": np.std(records[m], ddof=1) for m in y_metrics
        }

        stats["x_value"] = 0
        aggregated[label] = pd.DataFrame([stats])

    return aggregated


def aggregate_runs_with_stats(
    is_bayesian_opt: bool,
    runs: List[wandb.apis.public.Run],
    y_metrics: List[str],
    x_axis: str,
    num_runs: int,
    sort_metric: str = None,
    group_by: List[str] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Aggregates runs by solver (and optional grouping variables) to calculate mean and std.
    Selects the top 'num_runs' runs based on the specified metric for each group.

    Args:
        runs: List of wandb runs
        y_metrics: List of metrics to extract
        x_axis: X-axis for plotting ("time", "datapasses", or "iters")
        num_runs: Number of top runs to include for each group
        sort_metric: Metric used to select the best runs (default: first metric in y_metrics)
        group_by: Additional configuration keys to group by

    Returns:
        Dictionary with structure: {group_key: {"mean": mean_df, "std": std_df, "num_runs": actual_num_runs}}
    """
    if group_by is None:
        group_by = []

    if sort_metric is None and y_metrics:
        sort_metric = y_metrics[0]

    run_data = {}
    grouped_runs = defaultdict(list)

    for run in runs:
        try:
            df = get_run_history(
                is_bayesian_opt,
                run,
                [sort_metric] + [m for m in y_metrics if m != sort_metric],
                x_axis=x_axis,
            )

            if df.empty or sort_metric not in df.columns:
                print(
                    f"Warning: Run {run.id} missing sort metric {sort_metric}, skipping"
                )
                continue

            run_config = {
                CONFIG_KEYS["SOLVER"]: run.config.get(CONFIG_KEYS["SOLVER"], "unknown")
            }

            if run_config[CONFIG_KEYS["SOLVER"]] == "sap":
                solver_config = run.config.get("solver_config", {})
                run_config["precond_type"] = (
                    "nystrom" if "precond_config" in solver_config else "identity"
                )

            for key in group_by:
                config_key = CONFIG_KEYS.get(key, key)
                run_config[key] = run.config.get(config_key, "unknown")

            df["run_id"] = run.id
            for k, v in run_config.items():
                df[k] = v

            solver = run_config[CONFIG_KEYS["SOLVER"]]
            if solver == "sap" and "precond_type" in run_config:
                group_key = f"{solver}-{run_config['precond_type']}"
            else:
                group_key = solver

            for key in group_by:
                if key in run_config:
                    group_key += f"-{run_config[key]}"

            if sort_metric in {
                METRIC_PATHS["TEST_RMSE"],
                METRIC_PATHS["POSTERIOR_NLL"],
                METRIC_PATHS["POSTERIOR_MEAN_NLL"],
            }:
                score = df[sort_metric].iloc[-1]
                better_than = lambda x, y: x < y
            else:
                score = -df[sort_metric].iloc[-1]
                better_than = lambda x, y: x > y

            if not np.isfinite(score):
                print(
                    f"Warning: Run {run.id} has invalid {sort_metric} value: {score}, skipping"
                )
                continue

            run_data[run.id] = (score, df)
            grouped_runs[group_key].append((score, run.id))

        except Exception as e:
            print(f"Error processing run {run.id}: {str(e)}")

    aggregated_data = {}
    for group_key, runs_and_scores in grouped_runs.items():
        if not runs_and_scores:
            continue

        sorted_runs = sorted(runs_and_scores, key=lambda x: x[0])

        selected_run_ids = [run_id for _, run_id in sorted_runs[:num_runs]]
        actual_num_runs = len(selected_run_ids)

        if actual_num_runs == 0:
            print(f"Warning: No valid runs for group {group_key}")
            continue

        if actual_num_runs < num_runs:
            print(
                f"Warning: Group {group_key} has only {actual_num_runs} runs (requested {num_runs})"
            )

        selected_dfs = [
            run_data[run_id][1] for run_id in selected_run_ids if run_id in run_data
        ]

        if not selected_dfs:
            continue

        all_x = sorted(set().union(*[set(df["x_value"]) for df in selected_dfs]))

        aligned_dfs = []
        for df in selected_dfs:

            new_df = pd.DataFrame({"x_value": all_x})

            for metric in y_metrics:
                if metric in df.columns:
                    interp_vals = np.interp(
                        all_x,
                        df["x_value"],
                        df[metric],
                        left=df[metric].iloc[0],
                        right=df[metric].iloc[-1],
                    )
                    new_df[metric] = (
                        pd.Series(interp_vals).ffill().bfill()
                    )  # Fill any remaining NaNs
            aligned_dfs.append(new_df)

        mean_df = pd.DataFrame({"x_value": all_x})
        std_df = pd.DataFrame({"x_value": all_x})

        for metric in y_metrics:
            values = np.array(
                [df[metric].values for df in aligned_dfs if metric in df.columns]
            )
            if values.size == 0:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_df[metric] = np.nanmean(values, axis=0)
                std_df[metric] = np.nanstd(values, axis=0, ddof=1)

        solver_parts = group_key.split("-")
        solver = solver_parts[0]
        mean_df[CONFIG_KEYS["SOLVER"]] = solver
        std_df[CONFIG_KEYS["SOLVER"]] = solver

        if solver == "sap" and len(solver_parts) > 1:
            mean_df["sap_precond"] = solver_parts[1].capitalize()
            std_df["sap_precond"] = solver_parts[1].capitalize()

        if CONFIG_KEYS["DATASET"] in selected_dfs[0].columns:
            mean_df[CONFIG_KEYS["DATASET"]] = selected_dfs[0][
                CONFIG_KEYS["DATASET"]
            ].iloc[0]
            std_df[CONFIG_KEYS["DATASET"]] = selected_dfs[0][
                CONFIG_KEYS["DATASET"]
            ].iloc[0]

        aggregated_data[group_key] = {
            "mean": mean_df,
            "std": std_df,
            "num_runs": actual_num_runs,
        }

    return aggregated_data
