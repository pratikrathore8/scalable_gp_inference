import os
from pathlib import Path
import pandas as pd
from uci_datasets import all_datasets, Dataset as UCIDataset

# Create data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def download_uci_datasets(split: int = 0,
                          dataset_names: list[str] | None = None):
    """
    Downloads all the UCI datasets and splits each one into test and train datasets,
    then saves each one in a separate directory in DATA_DIR / dataset_name.

    Args:
        split (optional): The split index of the dataset. Defaults to 0.

        dataset_name (optional): If dataset_names is None, download *all* UCI datasets;
        otherwise only download the named ones. Defaults to None.
    """

    if dataset_names is None:
        to_download = all_datasets.keys()
    else:
        to_download = []
        for name in dataset_names:
            if name not in all_datasets:
                raise ValueError(f"Unknown UCI dataset: {name}")
            to_download.append(name)

    for dataset_name in to_download:
        try:
            print(f"Downloading {dataset_name}...")
            dataset = UCIDataset(dataset_name)
            x_train, y_train, x_test, y_test = dataset.get_split(split)

            # Create dataset directory
            dataset_dir = DATA_DIR / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Save training data
            train_df = pd.DataFrame(x_train)
            train_df['label'] = y_train
            train_df.to_csv(dataset_dir / 'train.csv', index=False)

            # Save test data
            test_df = pd.DataFrame(x_test)
            test_df['label'] = y_test
            test_df.to_csv(dataset_dir / 'test.csv', index=False)

            print(f"Saved {dataset_name} to {dataset_dir}")
        except Exception as e:
            print(f"Failed to download {dataset_name}: {e}")


if __name__ == "__main__":
    download_uci_datasets()