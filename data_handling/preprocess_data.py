import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from uci_datasets_configs import DATA_DIR

def preprocess_dataset(
    dataset_name: str,
    test_split_ratio: float,
    target_rank: int,
    normalize: bool,
    normalization_method: str,
    device: str,
    random_state: int = 42
) -> dict:

    """
    Preprocesses the datasets in DATA_DIR / dataset_name and returns tensors directly in memory

    Args:
        dataset_name: Short name of dataset from DATASET_CONFIGS

        test_split_ratio: Proportion for the test to train split (0-1)

        label_rank: Controls y's dimensionality for both the train and test datas.
          We either reshape y to be a column vector (shape (N, 1)) or squeeze the labels
          to become rank 1 arrays (shape (N,)). Some frameworks (e.g., PyTorch) might prefer targets
           as 2D tensors for batch processing, so here it could be set to 2.

        normalize: Whether to apply normalization or not

        normalization_method: Only z-score is included here

        device: Target device for tensors ('cpu' or 'cuda')

        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
        - x_train, y_train: Training data tensors
        - x_test, y_test: Test data tensors
        - norm_params: Normalization parameters (if applied)
    """
    dataset_dir = DATA_DIR / dataset_name

    try:
        # Load the full dataframe
        full_df = pd.read_csv(dataset_dir / f'{dataset_name}_df.csv')

    except FileNotFoundError:
        raise ValueError(f"Dataset {dataset_name} not found in {DATA_DIR}")


    # Manage and validate columns
    set_column_roles(dataset_name)
    config = DATASET_CONFIGS[dataset_name]
    target_columns = ['target']
    ignore_columns = config.get("ignore_columns", [])
    exclude_columns = target_columns + ignore_columns
    exclude_columns = [str(col) for col in exclude_columns] # Convert all of the header labels to strings to be sure

    for col in exclude_columns:
        if col not in full_df.columns:
            raise ValueError(f"Column '{col}' missing in {dataset_name} data")


    # Split data
    x_features = full_df.drop(columns=exclude_columns)
    y = full_df[target_columns]

    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y, test_size=test_split_ratio, random_state=42
    )

    # Separate and reshape features and labels
    x_train = x_train.values.astype(np.float32)
    y_train = y_train.values.astype(np.float32)
    x_test = x_test.values.astype(np.float32)
    y_test = y_test.values.astype(np.float32)

    if target_rank == 2:
        y_train = y_train.reshape(-1, 1)
        y_test  = y_test.reshape(-1, 1)
    else:
        y_train = y_train.squeeze()
        y_test  = y_test.squeeze()

    # Initialize normalization parameters
    normalization_params = None
    x_scaler, y_scaler = None, None

    if normalize:
        if normalization_method == 'z_score':
            # Feature normalization
            x_scaler = StandardScaler()
            x_train_proc = x_scaler.fit_transform(x_train)
            x_test_proc = x_scaler.transform(x_test)

            # Target normalization
            y_scaler = StandardScaler()
            y_train_proc = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test_proc = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

            # Store parameters
            normalization_params = {
                'x_scaler': x_scaler,
                'y_scaler': y_scaler
            }
        else:
            print(f"Unsupported normalization method: {normalization_method}")


    # Convert to tensors and move to device
    data_dict = {
        'x_train': torch.as_tensor(x_train_proc, device=device),
        'y_train': torch.as_tensor(y_train_proc, device=device),
        'x_test': torch.as_tensor(x_test_proc, device=device),
        'y_test': torch.as_tensor(y_test_proc, device=device),
        'normalization_params': normalization_params
    }

    print(f"Number of training samples for {dataset_name}: {len(data_dict['x_train'])}")
    print(f"Number of test samples for {dataset_name}: {len(data_dict['x_test'])}")
    print(f"Device: {data_dict['x_train'].device}")

    return data_dict

# Example usage
if __name__ == "__main__":
    data = preprocess_dataset(
        "3droad",
        0.2,
        2,
        True,
        'z_score',
        device="cuda" if torch.cuda.is_available() else "cpu")