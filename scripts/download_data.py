from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from src.machineguard.config import PROJECT_ROOT


DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
)

RAW_DATA_PATH = PROJECT_ROOT / "data/raw/ai4i2020.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data/processed/machine_data.csv"

COLUMN_MAPPING = {
    "Type": "type",
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
}

MODEL_COLUMNS = [
    "type",
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "machine_failure",
]


def download_dataset() -> Path:
    """Download the AI4I dataset from UCI."""
    RAW_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if RAW_DATA_PATH.exists():
        print(f"Raw dataset already exists: {RAW_DATA_PATH}")
        return RAW_DATA_PATH

    print("Downloading AI4I 2020 dataset...")
    urlretrieve(DATASET_URL, RAW_DATA_PATH)
    print(f"Dataset downloaded to: {RAW_DATA_PATH}")

    return RAW_DATA_PATH


def prepare_dataset(raw_path: Path) -> Path:
    """Rename and select production model columns."""
    dataframe = pd.read_csv(raw_path)

    missing_columns = [
        column for column in COLUMN_MAPPING if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

    processed = dataframe.rename(columns=COLUMN_MAPPING)

    processed = processed[MODEL_COLUMNS].copy()

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    print(f"Processed dataset saved to: {PROCESSED_DATA_PATH}")
    print(f"Dataset shape: {processed.shape}")
    print(f"Machine failure rate: {processed['machine_failure'].mean():.4f}")

    return PROCESSED_DATA_PATH


def main() -> None:
    """Download and prepare the training dataset."""
    raw_path = download_dataset()
    prepare_dataset(raw_path)


if __name__ == "__main__":
    main()
