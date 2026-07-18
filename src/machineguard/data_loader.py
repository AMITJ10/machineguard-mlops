from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_processed_data(
    data_path: Path,
) -> pd.DataFrame:
    """Load the processed machine dataset.

    Args:
        data_path: Location of the processed CSV file.

    Returns:
        Loaded machine dataset.

    Raises:
        FileNotFoundError: If the dataset does not exist.
        ValueError: If the dataset is empty.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}")

    dataframe = pd.read_csv(data_path)

    if dataframe.empty:
        raise ValueError(f"Processed dataset is empty: {data_path}")

    return dataframe


def split_machine_data(
    dataframe: pd.DataFrame,
    target_column: str,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Split data into training, validation and test sets.

    The configured validation size represents a fraction of the complete
    dataset. For example, test_size=0.20 and validation_size=0.20 creates
    approximately a 60/20/20 train-validation-test split.

    Args:
        dataframe: Validated machine dataset.
        target_column: Name of the prediction target.
        test_size: Fraction reserved for testing.
        validation_size: Fraction reserved for validation.
        random_state: Reproducibility seed.

    Returns:
        X_train, X_validation, X_test,
        y_train, y_validation and y_test.
    """
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column not found: {target_column}")

    if test_size + validation_size >= 1.0:
        raise ValueError("test_size + validation_size must be less than 1.")

    X = dataframe.drop(
        columns=[target_column],
    )
    y = dataframe[target_column]

    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    relative_validation_size = validation_size / (1.0 - test_size)

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=relative_validation_size,
        stratify=y_train_validation,
        random_state=random_state,
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )
