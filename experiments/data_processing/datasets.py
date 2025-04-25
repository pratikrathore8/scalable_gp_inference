from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import requests
import zipfile

import pandas as pd
from sklearn.datasets import fetch_openml

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

    # @abstractmethod
    # def load(self, load_path: str, *args, **kwargs):
    #     """Load the dataset from the specified location."""
    #     pass

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


@dataclass(kw_only=True, frozen=False)
class SGDMLDataset(_BaseDataset):
    def _raw_download(self, save_path: str, molecule: str):
        """Download the dataset from SGDML and save it to the specified location."""
        url = f"{SGDML_URL_STEM}/{molecule}.npz"
        response = requests.get(url)
        with open(os.path.join(save_path, "data.npz"), "wb") as f:
            f.write(response.content)


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
