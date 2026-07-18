"""Unit tests for machine-data schema validation."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest

from src.machineguard.data_schema import validate_machine_data


def valid_dataframe() -> pd.DataFrame:
    """Return one valid machine-data record."""
    return pd.DataFrame(
        {
            "type": ["M"],
            "air_temperature": [298.1],
            "process_temperature": [308.6],
            "rotational_speed": [1551],
            "torque": [42.8],
            "tool_wear": [120],
            "machine_failure": [0],
        }
    )


def test_valid_dataset_passes() -> None:
    """A valid dataset should pass schema validation."""
    validated = validate_machine_data(valid_dataframe())

    assert len(validated) == 1
    assert validated.loc[0, "type"] == "M"
    assert validated.loc[0, "machine_failure"] == 0


def test_negative_rotational_speed_fails() -> None:
    """Negative rotational speed should fail validation."""
    dataframe = valid_dataframe()
    dataframe.loc[0, "rotational_speed"] = -1

    with pytest.raises(
        (
            pa.errors.SchemaError,
            pa.errors.SchemaErrors,
        )
    ):
        validate_machine_data(dataframe)


def test_invalid_machine_type_fails() -> None:
    """Unknown machine type should fail validation."""
    dataframe = valid_dataframe()
    dataframe.loc[0, "type"] = "UNKNOWN"

    with pytest.raises(
        (
            pa.errors.SchemaError,
            pa.errors.SchemaErrors,
        )
    ):
        validate_machine_data(dataframe)


def test_invalid_target_value_fails() -> None:
    """Target should contain only binary values."""
    dataframe = valid_dataframe()
    dataframe.loc[0, "machine_failure"] = 5

    with pytest.raises(
        (
            pa.errors.SchemaError,
            pa.errors.SchemaErrors,
        )
    ):
        validate_machine_data(dataframe)
