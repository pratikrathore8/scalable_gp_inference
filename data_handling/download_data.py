import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
from sklearn.model_selection import train_test_split
from io import BytesIO
import ssl
import certifi
import json
from urllib.request import urlopen
from uci_configs import DATA_DIR, UCI_BASE_URL, DATASET_CONFIGS


def get_metadata(dataset_name: str):
    """Fetch dataset metadata from UCI API."""
    dataset_id = DATASET_CONFIGS[dataset_name]["id"]
    api_url = f"https://archive.ics.uci.edu/api/dataset?id={dataset_id}"

    context = ssl.create_default_context(cafile=certifi.where())

    try:
        with urlopen(api_url, context=context) as response:
            return json.load(response)
    except Exception as e:
        print(f"Error fetching metadata for {dataset_name}: {e}")
        return None


def set_label_column(dataset_name: str):
    """Set label column from metadata."""
    metadata = get_metadata(dataset_name)
    if metadata:
        target_col = metadata["data"].get("target_col", None)
    else:
        raise ValueError("No metadata found!")

    DATASET_CONFIGS[dataset_name]["label_columns"] = target_col


def load_data_from_zip(dataset_dir):
    """Load data from downloaded zip file."""
    for file_path in dataset_dir.glob('**/*'):
        if file_path.is_file() and not file_path.name.startswith('.'):
            try:
                return pd.read_csv(file_path)
            except:
                try:
                    return pd.read_csv(file_path, delim_whitespace=True,
                                       header=None)
                except:
                    continue
    raise ValueError("No readable data files found in zip archive")


def create_dataframe(dataset_name: str, test_size: float):
    """Main function to process and split datasets."""
    dataset_dir = DATA_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Set label column from metadata
    set_label_col(dataset_name)
    label_columns = DATASET_CONFIGS[dataset_name]["label_columns"]

    if label_columns is None:
        raise ValueError(f"Label columns not found for {dataset_name}")

    config = DATASET_CONFIGS[dataset_name]

    # Load data
    if config["data_url"]:
        response = requests.get(config["data_url"])
        data = pd.read_csv(BytesIO(response.content))
    else:
        response = requests.get(config["download_url"])
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(dataset_dir)
        data = load_data_from_zip(dataset_dir)

    # Validate label column
    for label_column in label_columns:
        if label_column not in data.columns:
            raise ValueError(
                f"Label column '{label_column}' not found in data columns.")

    # Split data
    x = data.drop(columns=label_columns)
    y = data[label_columns]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42
    )

    # Save splits
    train_df = x_train.copy()
    train_df["label"] = y_train
    test_df = x_test.copy()
    test_df["label"] = y_test

    train_df.to_csv(dataset_dir / "train.csv", index=False)
    test_df.to_csv(dataset_dir / "test.csv", index=False)

    print(f"Processed {dataset_name} | Train: {len(train_df)} | Test: {len(test_df)}")
    return train_df, test_df
