"""Unit tests for machine feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from src.machineguard.features import MachineFeatureEngineer


def sample_features() -> pd.DataFrame:
    """Return a valid feature-engineering input."""
    return pd.DataFrame(
        {
            "air_temperature": [298.0],
            "process_temperature": [308.0],
            "rotational_speed": [1500],
            "torque": [40.0],
            "tool_wear": [100],
        }
    )


def test_temperature_difference() -> None:
    """Temperature difference should be process minus air temperature."""
    transformer = MachineFeatureEngineer()

    transformed = transformer.fit_transform(sample_features())

    assert transformed.loc[
        0,
        "temperature_difference",
    ] == pytest.approx(10.0)


def test_mechanical_power_is_positive() -> None:
    """Mechanical power should be positive for valid input."""
    transformer = MachineFeatureEngineer()

    transformed = transformer.fit_transform(sample_features())

    assert (
        transformed.loc[
            0,
            "mechanical_power",
        ]
        > 0
    )


def test_transform_preserves_row_count() -> None:
    """Feature engineering should not add or remove rows."""
    dataframe = pd.concat(
        [
            sample_features(),
            sample_features(),
        ],
        ignore_index=True,
    )

    transformed = MachineFeatureEngineer().fit_transform(dataframe)

    assert len(transformed) == len(dataframe)


def test_transform_does_not_modify_original_dataframe() -> None:
    """Feature engineering should not mutate its input."""
    dataframe = sample_features()
    original = dataframe.copy(deep=True)

    MachineFeatureEngineer().fit_transform(dataframe)

    pd.testing.assert_frame_equal(
        dataframe,
        original,
    )
