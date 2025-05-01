
"""
Download *all* run-level data from the public W&B project
    sketchy-opts/gp_inference_acsincome

Creates:
    wandb_download/<run_name>_<run_id>/history.csv
                                         config.json
                                         summary.json
"""

import json
from pathlib import Path
import pandas as pd
import tqdm
import wandb

DATASET_NAME = "acsincome"
PROJECT = f"sketchy-opts/gp_inference_{DATASET_NAME}"
OUT_DIR = Path(f"wandb_download_results/{DATASET_NAME}")

api = wandb.Api()
runs = api.runs(PROJECT)

OUT_DIR.mkdir(parents=True, exist_ok=True)

for run in tqdm.tqdm(runs, desc="Downloading runs"):

    run_dir = OUT_DIR / f"{run.name}_{run.id[:8]}"
    run_dir.mkdir(exist_ok=True)

    rows = list(run.scan_history())

    if rows:
        df = pd.DataFrame(rows)
        csv_path = run_dir / "history.csv"
        df.to_csv(csv_path, index=False)
    else:
        (run_dir / "history.csv").touch()

    cfg_path = run_dir / "config.json"
    with cfg_path.open("w") as fh:
        json.dump(run.config, fh, indent=2, sort_keys=True)

    summary_path = run_dir / "summary.json"
    with summary_path.open("w") as fh:
        json.dump(run.summary._json_dict, fh, indent=2, sort_keys=True)

print(f"✅ Finished. Data saved under {OUT_DIR.resolve()}")
