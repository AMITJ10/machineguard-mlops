"""Prepare the latest dataset for the retraining pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"


def download_latest_data() -> Path:
    """Confirm that the latest local dataset is available.

    Returns:
        Path to the available dataset.

    Raises:
        FileNotFoundError: If the dataset does not exist.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Training dataset was not found at "
            f"{DATA_PATH}. Add the dataset or configure "
            "cloud-object-storage downloading."
        )

    logger.info(
        "Latest training dataset is available at %s.",
        DATA_PATH,
    )

    return DATA_PATH


def main() -> None:
    """Run the data preparation step."""
    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s | %(levelname)s | %(name)s | %(message)s"),
    )

    data_path = download_latest_data()

    print(f"Dataset ready: {data_path}")


if __name__ == "__main__":
    main()
