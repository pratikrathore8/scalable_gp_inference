import numpy as np
import pandas as pd


def _ensure_float(data: np.ndarray):
    """
    Ensure that the data is of type float64.
    """
    if data.dtype != np.float64:
        data = data.astype(np.float64)
    return data


def _convert_to_numpy(data: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
    """
    Convert a pandas DataFrame, Series, or numpy array to a numpy array.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return _ensure_float(data.to_numpy())
    elif isinstance(data, np.ndarray):
        return _ensure_float(data)
    else:
        raise ValueError(
            "Unsupported data type. Must be DataFrame, Series, or ndarray."
        )


def _process_molecule(R: np.ndarray) -> np.ndarray:
    n_atoms = R.shape[1]
    X = np.sum((R[:, :, np.newaxis, :] - R[:, np.newaxis, :, :]) ** 2, axis=-1) ** 0.5
    X = X[:, np.triu_indices(n_atoms, 1)[0], np.triu_indices(n_atoms, 1)[1]] ** -1.0

    return X


def _convert_datetime_columns(
    df: pd.DataFrame, datetime_columns: list[int]
) -> pd.DataFrame:
    """
    Convert specified date/time columns in a DataFrame to numerical values.

    Args:
        df: pandas DataFrame to process (will be modified in-place)
        datetime_columns: List of column indices to convert from datetime to numerical.

    Returns:
        The modified DataFrame with specified date/time columns
          converted to numerical values
    """
    # Convert negative indices to positive if needed
    processed_indices = []
    for idx in datetime_columns:
        if idx < 0:
            idx = len(df.columns) + idx
        processed_indices.append(idx)

    # Get the actual column names from indices
    columns_to_convert = [
        df.columns[idx] for idx in processed_indices if 0 <= idx < len(df.columns)
    ]

    for col in columns_to_convert:
        try:
            # Convert to datetime
            df[col] = pd.to_datetime(df[col], errors="coerce")

            # Check if dates are all at midnight (no time component)
            has_time_component = False
            non_null = df[col].dropna()

            if len(non_null) > 0:
                has_time_component = any(
                    (t.hour != 0 or t.minute != 0 or t.second != 0)
                    for t in non_null.dt.time
                )

            # Convert to a numerical representation
            if has_time_component:
                # If there's a time component, use unix timestamp (seconds since epoch)
                # nanoseconds to seconds
                df[col] = df[col].astype("int64") // 10**9
            else:
                # If it's just dates (no time), use days since a reference date
                reference_date = pd.Timestamp("1970-01-01")
                df[col] = (df[col] - reference_date).dt.days

        except (TypeError, ValueError) as e:
            print(
                "Failed to convert column at index "
                f"{processed_indices[columns_to_convert.index(col)]}: {e}"
            )
            continue

    return df
