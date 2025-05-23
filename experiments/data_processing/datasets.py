from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import requests
import zipfile

import h5py
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch

from experiments.data_processing.dockstring_utils import smiles_to_fingerprint_arr
from experiments.data_processing.utils import (
    _standardize,
    _convert_to_numpy,
    _numpy_to_torch,
    _process_molecule,
    _convert_datetime_columns,
)

DOCKSTRING_DATASET_URL_STEM = "https://figshare.com/ndownloader/files/35948138"
DOCKSTRING_SPLIT_URL_STEM = "https://figshare.com/ndownloader/files/35948123"
DOCKSTRING_DATASET_FILE_NAME = "dockstring-dataset.tsv"
DOCKSTRING_SPLIT_FILE_NAME = "cluster_split.tsv"
SGDML_URL_STEM = "http://www.quantum-machine.org/gdml/data/npz"
UCI_URL_STEM = "https://archive.ics.uci.edu/static/public"


@dataclass(kw_only=True, frozen=True)
class SplitData:
    Xtr: torch.Tensor
    Xtst: torch.Tensor
    ytr: torch.Tensor
    ytst: torch.Tensor


@dataclass(kw_only=True, frozen=False)
class _BaseDataset(ABC):
    name: str
    data_folder_name: str
    split_proportion: float | None = (
        None  # Unnecessary for downloading, needed for loading
    )
    split_shuffle: bool = True  # Unnecessary for downloading, needed for loading
    split_seed: int | None = None  # Unnecessary for downloading, needed for loading

    def _check_save_path(self, save_path: str):
        """Check if the save path exists and create it if it doesn't."""
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    @abstractmethod
    def _raw_download(self, save_path: str, *args, **kwargs):
        """Download the dataset and save it to the specified location."""
        pass

    def download(self, save_path: str, *args, **kwargs):
        """Download the dataset and save it to the specified location."""
        joined_save_path = os.path.join(save_path, self.data_folder_name)
        self._check_save_path(joined_save_path)
        self._raw_download(joined_save_path, *args, **kwargs)

    @abstractmethod
    def _raw_load(self, load_path: str, *args, **kwargs) -> dict:
        """Load the dataset from the specified location."""
        pass

    def _load(self, load_path: str, *args, **kwargs) -> dict[np.ndarray, np.ndarray]:
        """Load the dataset from the specified location as numpy arrays."""
        joined_load_path = os.path.join(load_path, self.data_folder_name)
        data = self._raw_load(joined_load_path, *args, **kwargs)
        for key, value in data.items():
            data[key] = _convert_to_numpy(value)
        return data

    def _split_data(
        self, data: dict[np.ndarray, np.ndarray]
    ) -> dict[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the data into training and testing sets.
        """
        Xtr, Xtst, ytr, ytst = train_test_split(
            data["X"],
            data["y"],
            test_size=self.split_proportion,
            shuffle=self.split_shuffle,
            random_state=self.split_seed,
        )
        return {
            "Xtr": Xtr,
            "Xtst": Xtst,
            "ytr": ytr,
            "ytst": ytst,
        }

    def _standardize_data(
        self, data: dict[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> dict[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Standardize the data.
        """
        Xtr, Xtst = _standardize(data["Xtr"], data["Xtst"])
        ytr, ytst = _standardize(data["ytr"], data["ytst"])
        return {
            "Xtr": Xtr,
            "Xtst": Xtst,
            "ytr": ytr,
            "ytst": ytst,
        }

    def _convert_to_torch(
        self,
        data: dict[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert the data to PyTorch tensors.
        """
        return _numpy_to_torch(data, dtype, device)

    def load_torch(
        self,
        load_path: str,
        standardize: bool,
        dtype: torch.dtype,
        device: torch.device,
        *args,  # useful for load()
        **kwargs,  # useful for load()
    ) -> SplitData:
        # Load data
        data = self._load(load_path, *args, **kwargs)

        # Split and (potentially) standardize the data
        data_split = self._split_data(data)
        if standardize:
            data_split = self._standardize_data(data_split)

        # Convert to PyTorch tensors on the specified device
        data_split = self._convert_to_torch(data_split, dtype, device)
        return SplitData(**data_split)


@dataclass(kw_only=True, frozen=False)
class DockstringDataset(_BaseDataset):
    # Implementation is based on
    # https://github.com/AustinT/tanimoto-random-features-neurips23/blob/main/trf23/datasets/dockstring.py
    # and https://github.com/cambridge-mlg/sgd-gp/blob/main/scalable_gps/data.py
    target: str | None = None  # Needed for loading only
    input_dim: int | None = None  # Needed for loading only
    binarize: bool = False  # Needed for loading only
    mean_y: float | None = None  # Needed for loading only

    def _raw_download(self, save_path: str):
        """Download the Dockstring data and save it to the specified location."""
        data_url = f"{DOCKSTRING_DATASET_URL_STEM}/{DOCKSTRING_DATASET_FILE_NAME}"
        split_url = f"{DOCKSTRING_SPLIT_URL_STEM}/{DOCKSTRING_SPLIT_FILE_NAME}"
        data_response = requests.get(data_url)
        split_response = requests.get(split_url)
        with open(os.path.join(save_path, "data.tsv"), "wb") as f:
            f.write(data_response.content)
        with open(os.path.join(save_path, "split.tsv"), "wb") as f:
            f.write(split_response.content)

    def _raw_load(self, load_path: str):
        X = pd.read_csv(
            os.path.join(load_path, "data.tsv"),
            sep="\t",
        ).set_index("inchikey")
        splits = (
            pd.read_csv(
                os.path.join(load_path, "split.tsv"),
                sep="\t",
            )
            .set_index("inchikey")
            .loc[X.index]
        )
        return {"X": X, "splits": splits}

    def _load(self, load_path: str):
        # Don't convert to numpy as we usually do
        joined_load_path = os.path.join(load_path, self.data_folder_name)
        return self._raw_load(joined_load_path)

    def _split_data(self, data: dict[str, pd.DataFrame]):
        Xtr = data["X"][data["splits"]["split"] == "train"]
        Xtst = data["X"][data["splits"]["split"] == "test"]
        Xtr = Xtr[["smiles", self.target]].dropna()  # Drop rows with NaN values
        Xtst = Xtst[["smiles", self.target]].dropna()  # Drop rows with NaN values

        # Extract train/test SMILES
        ytr = _convert_to_numpy(Xtr[self.target])
        ytst = _convert_to_numpy(Xtst[self.target])
        Xtr = Xtr.smiles.to_list()
        Xtst = Xtst.smiles.to_list()

        # Clip targets to max of 5.0
        ytr = np.minimum(ytr, 5.0)
        ytst = np.minimum(ytst, 5.0)

        fp_kwargs = dict(
            use_counts=True, radius=1, nbits=self.input_dim, binarize=self.binarize
        )
        Xtr = smiles_to_fingerprint_arr(Xtr, **fp_kwargs).astype(np.float64)
        Xtst = smiles_to_fingerprint_arr(Xtst, **fp_kwargs).astype(np.float64)

        return {"Xtr": Xtr, "Xtst": Xtst, "ytr": ytr, "ytst": ytst}

    def _standardize_data(
        self, data: dict[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ):
        # Don't normalize the hashed features for the Dockstring datasets
        # We use mean_y to standardize the targets
        data["ytr"] = data["ytr"] - self.mean_y
        data["ytst"] = data["ytst"] - self.mean_y
        return data


@dataclass(kw_only=True, frozen=False)
class OpenMLDataset(_BaseDataset):
    def _raw_download(self, save_path: str, id: int):
        """Download the dataset from OpenML and save it to the specified location."""
        data, target = fetch_openml(data_id=id, return_X_y=True)
        pd.to_pickle(data, os.path.join(save_path, "data.pkl"))
        pd.to_pickle(target, os.path.join(save_path, "target.pkl"))

    def _raw_load(self, load_path: str):
        X = pd.read_pickle(os.path.join(load_path, "data.pkl"))
        y = pd.read_pickle(os.path.join(load_path, "target.pkl"))
        return {"X": X, "y": y}


@dataclass(kw_only=True, frozen=False)
class SGDMLDataset(_BaseDataset):
    def _raw_download(self, save_path: str, molecule: str):
        """Download the dataset from SGDML and save it to the specified location."""
        url = f"{SGDML_URL_STEM}/{molecule}.npz"
        response = requests.get(url)
        with open(os.path.join(save_path, "data.npz"), "wb") as f:
            f.write(response.content)

    def _raw_load(self, load_path: str):
        data = np.load(os.path.join(load_path, "data.npz"))
        X = _process_molecule(data["R"])
        y = np.squeeze(data["E"])
        return {"X": X, "y": y}


@dataclass(kw_only=True, frozen=False)
class TaxiDataset(_BaseDataset):
    def _raw_download(self, save_path: str):
        raise NotImplementedError(
            "Taxi dataset is not available for download. "
            "Please provide the data manually."
        )

    def _raw_load(self, load_path: str):
        with h5py.File(os.path.join(load_path, "data.h5py"), "r") as f:
            X, y = f["X"][()], f["Y"][()]
        y = np.squeeze(y)
        return {"X": X, "y": y}


@dataclass(kw_only=True, frozen=False)
class UCIDataset(_BaseDataset):
    def _raw_download(self, save_path: str, id: int, uci_file_name: str):
        """Download the dataset from UCI, extract it, and clean up the zip file."""
        # Download the zip file
        url = f"{UCI_URL_STEM}/{id}/{uci_file_name}.zip"
        response = requests.get(url)
        zip_path = os.path.join(save_path, "data.zip")
        with open(zip_path, "wb") as f:
            f.write(response.content)

        # Unzip the file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(save_path)

        # Delete the zip file
        os.remove(zip_path)

        # NOTE(pratik): the uci datasets we download are all .txt files after unzipping
        # so we rename the .txt file to data.txt

        # Find and rename all .txt files to data.txt
        txt_files = [f for f in os.listdir(save_path) if f.endswith(".txt")]

        # If there's only one .txt file, rename it to data.txt
        if len(txt_files) == 1:
            os.rename(
                os.path.join(save_path, txt_files[0]),
                os.path.join(save_path, "data.txt"),
            )
        # If there are multiple .txt files, you might want to handle differently
        elif len(txt_files) > 1:
            # Option 1: Rename the first .txt file to data.txt
            os.rename(
                os.path.join(save_path, txt_files[0]),
                os.path.join(save_path, "data.txt"),
            )
            # Log a warning about multiple files
            print(
                f"Warning: Multiple .txt files found in {save_path}. "
                f"Renamed {txt_files[0]} to data.txt."
            )

    def _raw_load(
        self,
        load_path: str,
        target_column: int = -1,
        datetime_columns: list | None = None,
        drop_columns: list | None = None,
        skip_header: bool = False,
        delimiter: str | None = None,
    ):
        """
        Load the dataset from a text file.

        Args:
            load_path: Path to the directory containing the data.txt file
            target_column: Index of the target column (negative indexing allowed)
            datetime_columns: List of column indices to convert from datetime to numeric
            drop_columns: List of column indices to drop from the feature matrix.
            skip_header: Whether to skip the first row (usually for column headers)
            delimiter: Delimiter for the text file.
                If None, attempts to detect automatically

        Returns:
            Dictionary with 'X' and 'y' keys containing features and target
        """
        file_path = os.path.join(load_path, "data.txt")

        # # Try to automatically determine the delimiter if not provided
        if delimiter is None:
            with open(file_path, "r") as f:
                first_line = f.readline().strip()
                if "," in first_line:
                    delimiter = ","
                elif "\t" in first_line:
                    delimiter = "\t"
                elif ";" in first_line:
                    delimiter = ";"
                else:
                    delimiter = " "  # Default to space

        # Read the data with pandas
        data = pd.read_csv(
            file_path,
            delimiter=delimiter,
            header=None,
            engine="python",  # More flexible handling of delimiters
        )

        # Skip the first row if needed
        if skip_header:
            data = data.iloc[1:].reset_index(drop=True)

        # Remove rows with missing values
        data.dropna(inplace=True)
        # Reset index after dropping rows
        data.reset_index(drop=True, inplace=True)

        # Convert datetime columns to numeric if specified
        if datetime_columns is not None:
            data = _convert_datetime_columns(data, datetime_columns)

        # Convert negative target_column to positive index
        if target_column < 0:
            target_column = len(data.columns) + target_column

        # Extract the target column
        y = data.iloc[:, target_column]

        # Create a list of columns to keep (all except target and drop_columns)
        columns_to_drop = [target_column]

        # Process drop_columns if provided
        if drop_columns is not None:
            for col in drop_columns:
                # Convert negative indices to positive
                if col < 0:
                    col = len(data.columns) + col
                columns_to_drop.append(col)

        # Remove duplicates and sort in descending order to avoid index shifting
        columns_to_drop = sorted(set(columns_to_drop), reverse=True)

        # Keep only the columns we want for X
        keep_columns = [i for i in range(len(data.columns)) if i not in columns_to_drop]
        X = data.iloc[:, keep_columns]

        return {"X": X, "y": y}
