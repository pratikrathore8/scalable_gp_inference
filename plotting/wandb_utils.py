import math
from functools import lru_cache
from typing import Iterable, Optional, Dict, List
import pandas as pd
import wandb
import numpy as np


def get_project_runs(
    entity: str,
    project: str,
    filters: Optional[Dict] = None,
    max_runs: Optional[int] = None
) -> List[wandb.apis.public.Run]:
    """Fetch runs with optional filters and limit."""
    api = wandb.Api()
    try:
        runs = api.runs(f"{entity}/{project}", order="-created_at", filters=filters)
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
            n = config.get("ntr", 1)
            m = config.get("rf_config", {}).get("num_features", 0)
            eval_freq = config.get("eval_freq", 1)
            scaling = (2 * m * eval_freq) / n if m else eval_freq/n
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
    x_axis: str = "step"
) -> Dict[str, pd.DataFrame]:
    data = {}
    for run in runs:
        try:
            df = get_run_history(run, y_metrics, x_axis=x_axis)
            if not df.empty:
                df["run_id"] = run.id
                df["run_name"] = run.name
                data[run.id] = df
        except Exception as e:
            print(f"Critical error with run {run.id}: {str(e)}")
    return data


def filter_runs(
    runs: Iterable[wandb.apis.public.Run],
    require_all: Optional[Dict] = None,
    require_any: Optional[Dict] = None
) -> List[wandb.apis.public.Run]:
    """Flexible run filtering with multiple conditions."""
    filtered = []
    for run in runs:
        config = run.config
        
        # All must match
        if require_all and not all(config.get(k) == v for k, v in require_all.items()):
            continue
            
        # Any must match
        if require_any and not any(config.get(k) == v for k, v in require_any.items()):
            continue
            
        filtered.append(run)
    return filtered

