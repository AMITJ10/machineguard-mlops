"""Generate shifted production-like data for drift testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_PATH = PROJECT_ROOT / "data" / "reference" / "reference.csv"

CURRENT_PATH = PROJECT_ROOT / "data" / "current" / "current.csv"

RANDOM_SEED = 42
MAX_SAMPLE_SIZE = 1_000


def simulate_drift(
    reference_path: Path = REFERENCE_PATH,
    current_path: Path = CURRENT_PATH,
) -> pd.DataFrame:
    """Create a shifted production dataset.

    Args:
        reference_path: Reference dataset path.
        current_path: Output path for shifted data.

    Returns:
        Shifted production-like dataframe.

    Raises:
        FileNotFoundError: If reference data is missing.
        ValueError: If data is empty or required columns
            are unavailable.
    """
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference dataset was not found: {reference_path}")

    reference = pd.read_csv(reference_path)

    if reference.empty:
        raise ValueError("Reference dataset is empty.")

    required_columns = {
        "rotational_speed",
        "torque",
        "tool_wear",
    }

    missing_columns = required_columns.difference(reference.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))

        raise ValueError(f"Reference dataset is missing required columns: {missing}")

    sample_size = min(
        MAX_SAMPLE_SIZE,
        len(reference),
    )

    current = reference.sample(
        n=sample_size,
        random_state=RANDOM_SEED,
    ).copy()

    random_generator = np.random.default_rng(RANDOM_SEED)

    # Simulate machines operating at much
    # higher rotational speeds.
    current["rotational_speed"] = (
        current["rotational_speed"]
        + random_generator.normal(
            loc=500,
            scale=100,
            size=len(current),
        )
    ).clip(lower=0)

    # Simulate increased mechanical load.
    current["torque"] = (
        current["torque"]
        * random_generator.normal(
            loc=1.25,
            scale=0.05,
            size=len(current),
        )
    ).clip(lower=0)

    # Simulate older tools in production.
    current["tool_wear"] = (
        current["tool_wear"]
        + random_generator.normal(
            loc=30,
            scale=8,
            size=len(current),
        )
    ).clip(lower=0)

    current_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    current.to_csv(
        current_path,
        index=False,
    )

    return current


def main() -> None:
    """Generate and save shifted production data."""
    current = simulate_drift()

    display_path = CURRENT_PATH.relative_to(PROJECT_ROOT)

    print("Shifted production batch saved successfully.")

    print(f"Path: {display_path}")

    print(f"Rows: {len(current)}")


if __name__ == "__main__":
    main()
