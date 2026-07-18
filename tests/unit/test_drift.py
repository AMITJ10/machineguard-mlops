"""Tests for production drift simulation."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.simulate_production_data import simulate_drift


def test_simulate_drift_creates_output(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"

    reference = pd.DataFrame(
        {
            "rotational_speed": [1_000, 1_100, 1_200],
            "torque": [30.0, 35.0, 40.0],
            "tool_wear": [10.0, 20.0, 30.0],
        }
    )

    reference.to_csv(
        reference_path,
        index=False,
    )

    current = simulate_drift(
        reference_path=reference_path,
        current_path=current_path,
    )

    assert current_path.exists()
    assert len(current) == len(reference)
    assert current["rotational_speed"].mean() > reference["rotational_speed"].mean()


def test_simulate_drift_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        simulate_drift(
            reference_path=tmp_path / "missing.csv",
            current_path=tmp_path / "current.csv",
        )
