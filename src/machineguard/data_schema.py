import pandas as pd
import pandera.pandas as pa


MACHINE_DATA_SCHEMA = pa.DataFrameSchema(
    {
        "type": pa.Column(
            str,
            checks=pa.Check.isin(["L", "M", "H"]),
            nullable=False,
        ),
        "air_temperature": pa.Column(
            float,
            checks=pa.Check.in_range(250.0, 350.0),
            nullable=False,
        ),
        "process_temperature": pa.Column(
            float,
            checks=pa.Check.in_range(250.0, 400.0),
            nullable=False,
        ),
        "rotational_speed": pa.Column(
            int,
            checks=pa.Check.in_range(0, 10_000),
            nullable=False,
        ),
        "torque": pa.Column(
            float,
            checks=pa.Check.in_range(0.0, 200.0),
            nullable=False,
        ),
        "tool_wear": pa.Column(
            int,
            checks=pa.Check.in_range(0, 1_000),
            nullable=False,
        ),
        "machine_failure": pa.Column(
            int,
            checks=pa.Check.isin([0, 1]),
            nullable=False,
        ),
    },
    strict=True,
    coerce=True,
)


def validate_machine_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Validate machine data against the production schema."""
    validated_dataframe = MACHINE_DATA_SCHEMA.validate(
        dataframe,
        lazy=True,
    )

    duplicate_count = int(
        validated_dataframe.duplicated().sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Dataset contains {duplicate_count} duplicate rows."
        )

    return validated_dataframe