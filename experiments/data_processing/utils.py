import numpy as np
import pandas as pd


def _convert_to_numpy(data: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
    """
    Convert a pandas DataFrame, Series, or numpy array to a numpy array.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise ValueError(
            "Unsupported data type. Must be DataFrame, Series, or ndarray."
        )


def _process_molecule(R: np.ndarray) -> np.ndarray:
    n_atoms = R.shape[1]
    X = np.sum((R[:, :, np.newaxis, :] - R[:, np.newaxis, :, :]) ** 2, axis=-1) ** 0.5
    X = X[:, np.triu_indices(n_atoms, 1)[0], np.triu_indices(n_atoms, 1)[1]] ** -1.0

    return X
