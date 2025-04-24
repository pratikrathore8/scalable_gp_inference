from .configs import DATA_DIR, DATASET_CONFIGS
from .utils import create_dataframe


def download_dataset(dataset_name: str, *, force: bool = False):
    """
    Download and save one dataset as a DataFrame in a csv file.

    Args:
        dataset_name : str
        force : bool
               Re-download even if the processed CSV already exists.
    """
    dataset_path = DATA_DIR / dataset_name / f"{dataset_name}_df.csv"
    if not force and dataset_path.exists():
        print(f"The {dataset_name} dataset already exists in {dataset_path}")
        return
    return create_dataframe(dataset_name)


def download_all_datasets(dataset_names=None, *, force: bool = False):
    """
    Loop over the registry and make sure every dataset is ready.

    Args:
        dataset_names : list[str] | None
                       Subset of dataset names to process; None → all available.
                       Defaulted to None (so it downloads all)
        force : bool
               Re-download even if the processed CSV already exists.
    """
    dataset_names = dataset_names or DATASET_CONFIGS.keys()
    for dataset_name in dataset_names:
        print(f"→ Downloading {dataset_name}")
        download_dataset(dataset_name, force=force)


if __name__ == "__main__":
    download_all_datasets()
