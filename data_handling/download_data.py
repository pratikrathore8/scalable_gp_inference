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
from uci_configs import DATASET_CONFIGS


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


def set_target_column(dataset_name: str):
    """Set target column from metadata."""
    metadata = get_metadata(dataset_name)
    if metadata:
        target_col = metadata["data"].get("target_col", None)
    else:
        raise ValueError("No metadata found!")

    DATASET_CONFIGS[dataset_name]["target_columns"] = target_col


def set_column_roles(dataset_name: str):
    """Set target and ignore columns from metadata based on variable roles."""
    if DATASET_CONFIGS[dataset_name]["target_columns"] is None:
        metadata = get_metadata(dataset_name)
        if not metadata:
            raise ValueError(f"No metadata found for {dataset_name}")

        variables = metadata.get("data", {}).get("variables", [])
        target_columns = []
        ignore_columns = []
        roles_to_ignore = {'ID', 'Other',
                           'Ignore'}  # Define which roles to exclude

        for var in variables:
            var_name = var.get("name", "").strip()
            if not var_name:
                continue

            role = var.get("role", "").strip()
            if role == 'Target':
                target_columns.append(var_name)
            elif role in roles_to_ignore:
                ignore_columns.append(var_name)

        # Update dataset configuration
        DATASET_CONFIGS[dataset_name]["target_columns"] = target_columns
        DATASET_CONFIGS[dataset_name]["ignore_columns"] = ignore_columns

        if not target_columns:
            raise ValueError(f"No target columns found for {dataset_name}")


def columns_are_numeric(cols):
    """check if all column names can be converted to numeric values.
     If true, it indicates the data lacks headers."""
    for col in cols:
        try:
            float(col)
        except ValueError:
            return False
    return True


def create_dataframe(dataset_name: str, test_size: float):
    """Process dataset while excluding non-feature columns."""
    dataset_dir = DATA_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Get column roles from metadata
    set_column_roles(dataset_name)
    config = DATASET_CONFIGS[dataset_name]
    target_columns = config["target_columns"]
    ignore_columns = config.get("ignore_columns", [])

    # Load data
    if config["data_url"]:
        response = requests.get(config["data_url"])
        data = pd.read_csv(BytesIO(response.content))
    else:
        response = requests.get(config["download_url"])
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(dataset_dir)
        data = load_data_from_zip(dataset_dir)

    # Check if columns are numeric strings and reset if necessary
    if columns_are_numeric(data.columns.astype(str)):
        new_row = data.columns.to_numpy()
        data.columns = range(data.shape[1])
        data = pd.DataFrame(
            np.vstack([new_row, data.to_numpy()]),
            columns=data.columns
        )

    # Validate columns
    exclude_columns = target_columns + ignore_columns
    for col in exclude_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' missing in {dataset_name} data")

    # Split data
    x = data.drop(columns=exclude_columns)
    y = data[target_columns]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42
    )

    # Save splits
    train_df = x_train.copy()
    train_df["target"] = y_train
    test_df = x_test.copy()
    test_df["target"] = y_test

    train_df.to_csv(dataset_dir / "train.csv", index=False)
    test_df.to_csv(dataset_dir / "test.csv", index=False)

    print(
        f"Processed {dataset_name} | Train: {len(train_df)} | Test: {len(test_df)}")
    return train_df, test_df


# Example usage
if __name__ == "__main__":
    create_dataframe("song", test_size=0.1)