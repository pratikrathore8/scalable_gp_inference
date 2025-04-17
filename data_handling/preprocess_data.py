import numpy as np
import json
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
PROCESSED_DIR = Path("processed_data")
PROCESSED_DIR.mkdir(exist_ok=True)


def normalize_data_z_score(X_train, X_test, y_train, y_test):
    """
    Normalize data using z-score normalization
    """
    # Feature normalization
    mu_x = np.mean(X_train, axis=0)
    sigma_x = np.std(X_train, axis=0)
    sigma_x[sigma_x == 0] = 1.0

    X_train_norm = (X_train - mu_x) / sigma_x
    X_test_norm = (X_test - mu_x) / sigma_x

    # Label normalization
    mu_y = np.mean(y_train)
    sigma_y = np.std(y_train)
    sigma_y = 1.0 if sigma_y == 0 else sigma_y

    y_train_norm = (y_train - mu_y) / sigma_y
    y_test_norm = (y_test - mu_y) / sigma_y

    return (X_train_norm, X_test_norm, y_train_norm, y_test_norm,
            mu_x, sigma_x, mu_y, sigma_y)


def preprocess_datasets(normalize: bool = True,
                        normalization_method: str = 'z_score', label_rank: int = 2):
    """
    Preprocesses the datasets in DATA_DIR / dataset_name, saving the preprocessed data in
     new directories PROCESSED_DIR / dataset_name.

      Args:
          normalization: Normalize the preprocessed data or not. Defaults to True.

          normalization_method: Defaults to z score.

          label_rank: Controls y's dimensionality for both the train and test datas.
          We wither reshape y to be a column vector (shape (N, 1)) or squeeze the labels
          to become rank 1 arrays (shape (N,)). Some frameworks (e.g., PyTorch) might prefer targets
           as 2D tensors for batch processing, so here it defaults to 2.
    """
    for dataset_path in DATA_DIR.iterdir():
        if not dataset_path.is_dir():
            continue

        dataset_name = dataset_path.name
        print(f"Preprocessing {dataset_name}...")

        try:
            # Load data
            train_df = pd.read_csv(dataset_path / 'train.csv')
            test_df = pd.read_csv(dataset_path / 'test.csv')

            # Separate and reshape features and labels
            X_train = train_df.drop(columns=['label']).values.astype(np.float64)
            y_train = train_df['label'].values.astype(np.float64)
            X_test = test_df.drop(columns=['label']).values.astype(np.float64)
            y_test = test_df['label'].values.astype(np.float64)

            if label_rank == 1:
                y_train = y_train.squeeze()
                y_test  = y_test.squeeze()

            elif label_rank == 2:
                y_train = y_train.reshape(-1, 1)
                y_test  = y_test.reshape(-1, 1)

            # Apply normalization if requested
            if normalize:
                if normalization_method == 'z_score':
                    (X_train_proc, X_test_proc, y_train_proc, y_test_proc,
                     mu_x, sigma_x, mu_y, sigma_y) = normalize_data_z_score(
                        X_train, X_test, y_train, y_test)
                else:
                    print(
                        f"Normalization method {normalization_method} not supported")

            else:
                X_train_proc, X_test_proc = X_train, X_test
                y_train_proc, y_test_proc = y_train, y_test

            # Create processed directory
            processed_dir = PROCESSED_DIR / dataset_name
            processed_dir.mkdir(parents=True, exist_ok=True)

            # Save processed data
            np.save(processed_dir / 'X_train.npy', X_train_proc)
            np.save(processed_dir / 'y_train.npy', y_train_proc)
            np.save(processed_dir / 'X_test.npy', X_test_proc)
            np.save(processed_dir / 'y_test.npy', y_test_proc)

            # Save normalization parameters if normalized (only supports z_score at the moment)
            if normalize and normalization_method == 'z_score':
                params = {
                    'mu_x': mu_x.tolist(),
                    'sigma_x': sigma_x.tolist(),
                    'mu_y': mu_y.item(),
                    'sigma_y': sigma_y.item()
                }
                with open(processed_dir / 'params.json', 'w') as f:
                    json.dump(params, f)

            print(f"Processed {dataset_name} saved to {processed_dir}")
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")


if __name__ == "__main__":
    preprocess_datasets(normalize=True, normalization_method= 'z_score', label_rank= 2)