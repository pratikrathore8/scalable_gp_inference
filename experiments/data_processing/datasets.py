from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import requests
import zipfile

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from experiments.data_processing.utils import _convert_to_numpy, _process_molecule

SGDML_URL_STEM = "http://www.quantum-machine.org/gdml/data/npz"


@dataclass(kw_only=True, frozen=False)
class _BaseDataset(ABC):
    name: str
    data_folder_name: str
    train_proportion: float | None = None  # Unnecessary for downloading
    loading_seed: int | None = None  # Unnecessary for downloading

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
    def _raw_load(self, load_path: str, *args, **kwargs):
        """Load the dataset from the specified location."""
        pass

    def load(self, load_path: str, *args, **kwargs):
        """Load the dataset from the specified location."""
        joined_load_path = os.path.join(load_path, self.data_folder_name)
        data = self._raw_load(joined_load_path, *args, **kwargs)
        for key, value in data.items():
            data[key] = _convert_to_numpy(value)
        return data

    # The load function should read data from memory and also
    #  drop the appropriate features

    # TODO(pratik): add more functions for actually loading the data into memory
    # This should take care of things like splitting/shuffling data, dropping features,
    # preprocessing molecule data, etc.


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
class UCIDataset(_BaseDataset):
    def _raw_download(self, save_path: str, id: int, uci_file_name: str):
        """Download the dataset from UCI, extract it, and clean up the zip file."""
        # Download the zip file
        url = f"https://archive.ics.uci.edu/static/public/{id}/{uci_file_name}.zip"
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
        skip_header: bool = False,
        delimiter: str = None,
    ):
        """
        Load the dataset from a text file.

        Args:
            load_path: Path to the directory containing the data.txt file
            target_column: Index of the target column (negative indexing allowed)
            skip_header: Whether to skip the first row (usually for column headers)
            delimiter: Delimiter for the text file.
            If None, attempts to detect automatically

        Returns:
            Dictionary with 'X' and 'y' keys containing features and target
        """
        file_path = os.path.join(load_path, "data.txt")

        # Try to automatically determine the delimiter if not provided
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
            header=0 if skip_header else None,
            engine="python",  # More flexible handling of delimiters
        )

        # Extract X and y
        if target_column < 0:
            target_column = len(data.columns) + target_column

        y = data.iloc[:, target_column]
        X = data.drop(data.columns[target_column], axis=1)

        return {"X": X, "y": y}
