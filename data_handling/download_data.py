import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import tarfile
from io import BytesIO
import ssl
import certifi
import json
from urllib.request import urlopen
import subprocess
import bz2
import lzma
import shutil
from scipy.io import savemat
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from dataset_configs import DATA_DIR, DATASET_CONFIGS


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


def print_compressed_members(zip_bytes: bytes, extract_root: Path):
    """Echo every path inside buzz ZIP and the nested tar.gz."""
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            print("ZIP member →", name)             # first-layer listing
            if name.endswith(".tar.gz"):
                tar_bytes = zf.read(name)
                with tarfile.open(fileobj=BytesIO(tar_bytes), mode="r:gz") as tf:
                    for m in tf.getmembers():
                        print("TAR member →", m.name)  # second-layer listing
                    # unpack the tarball so files exist on disk
                    tf.extractall(extract_root)
                    print("Extracted TAR →", extract_root)
                    

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


def make_datetime_numeric(df, precision, date_col="Date", time_col="Time",
                           drop_original=True):

    dt = pd.to_datetime(df[date_col] + " " + df[time_col],
                        format="%d/%m/%Y %H:%M:%S",
                        errors="coerce")
    df = df.loc[dt.notna()].copy()
    dt = dt.loc[dt.notna()]

    # convert to seconds since epoch and store as float32
    df["timestamp"] = dt.astype("int64") // 10**9   # seconds
    df["timestamp"] = df["timestamp"].astype(precision)

    if drop_original:
        df = df.drop(columns=[date_col, time_col])

    return df


def columns_are_numeric(cols):
    """check if all column names can be converted to numeric values.
     If true, it indicates the data lacks headers."""
    for col in cols:
        try:
            float(col)
        except ValueError:
            return False
    return True


def load_buzz_regression(zip_bytes: bytes, dataset_dir) -> pd.DataFrame:
    """Return a tidy DF for Buzz regression, unpacking regression.tar.gz."""
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        tar_bytes = zf.read("regression.tar.gz")
    # unpack tar into dataset_dir/regression/
    with tarfile.open(fileobj=BytesIO(tar_bytes), mode="r:gz") as tf:
        tf.extractall(dataset_dir)                  # creates regression/Twitter/…
        # iterate .data files
        dfs = []
        for member in tf.getmembers():
            if member.name.endswith(".data"):
                part_path = dataset_dir / member.name
                src = "twitter" if "Twitter" in member.name else "tomshw"
                df_part = pd.read_csv(part_path, header=None, dtype=np.float32)
                df_part["source"] = src
                dfs.append(df_part)
    if not dfs:
        raise ValueError("No .data files inside regression.tar.gz")
    raw_data = pd.concat(dfs, ignore_index=True)
    raw_data = raw_data.dropna(axis=1, how='any')
    twitter_rows = raw_data[raw_data["source"] == "twitter"]
    data = twitter_rows.drop(columns=["source"])
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    df = x.copy()
    df["target"] = y
    return df



def create_dataframe(dataset_name: str):
    """Download the dataset, organize and assign the header labels, and save as a dataframe"""
    dataset_dir = DATA_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    config = DATASET_CONFIGS[dataset_name]
    source = config["source"]
    num_instances = config["num_instances"]
    num_features = config["num_features"]

    if source == "openml":
        data, target = fetch_openml(data_id=config["id"], return_X_y=True,
                                    as_frame=True)
        df = pd.DataFrame(data)
        df["target"] = target
        df.to_csv(dataset_dir / f"{dataset_name}_df.csv", index=False)

        print(
            f"Processed {dataset_name} | Number of instances: {num_instances} | Number of features: {num_features}")

        return df

    elif source == "sgdml":
        response = requests.get(config["download_url"])
        npz_data = np.load(BytesIO(response.content))

        # Process molecule (from fast_krr github repo: _process_molecule)
        R = npz_data["R"]
        X = np.sum((R[:, :, np.newaxis, :] - R[:, np.newaxis, :, :]) ** 2,
                   axis=-1) ** 0.5
        X = X[:, np.triu_indices(R.shape[1], 1)[0],
            np.triu_indices(R.shape[1], 1)[1]] ** -1.0

        y = npz_data["E"].squeeze()
        df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        df["target"] = y
        df.to_csv(dataset_dir / f"{dataset_name}_df.csv", index=False)

        print(
            f"Processed {dataset_name} | Number of instances: {num_instances} | Number of features: {num_features}")

        return df

    elif dataset_name == "buzz":
        response = requests.get(config["download_url"])
        print_compressed_members(response.content, dataset_dir)
        df = load_buzz_regression(response.content, dataset_dir)
        df.to_csv(dataset_dir / f"{dataset_name}_df.csv", index=False)
        print(
            f"Processed {dataset_name} | Number of instances: {num_instances} | Number of features: {num_features}")
        return df

    else:
        # Get column roles from metadata
        set_column_roles(dataset_name)
        target_columns = config["target_columns"]
        ignore_columns = config.get("ignore_columns", [])

        # Load data
        if config["data_url"]:
            response = requests.get(config["data_url"])
            data = pd.read_csv(BytesIO(response.content), na_values=['?'])
            data = data.dropna()
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

        # create full dataframe
        x = data.drop(columns=target_columns)
        y = data[target_columns]

        df = x.copy()
        df["target"] = y

        df.to_csv(dataset_dir / f"{dataset_name}_df.csv", index=False)

        print(
            f"Processed {dataset_name} | Number of instances: {num_instances} | Number of features: {num_features}")
        print(f"Columns to drop: {ignore_columns}")

        return df


if __name__ == "__main__":
   df = create_dataframe("buzz")