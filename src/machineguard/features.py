from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MachineFeatureEngineer(
    BaseEstimator,
    TransformerMixin,
):
    """Create domain-informed predictive-maintenance features."""

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> "MachineFeatureEngineer":
        """Learn transformer parameters.

        This transformer does not require learned parameters.
        """
        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create additional machine-condition features."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MachineFeatureEngineer expects a pandas DataFrame.")

        required_columns = {
            "air_temperature",
            "process_temperature",
            "rotational_speed",
            "torque",
            "tool_wear",
        }

        missing_columns = required_columns.difference(X.columns)

        if missing_columns:
            raise ValueError(
                "Missing columns required for feature engineering: "
                f"{sorted(missing_columns)}"
            )

        transformed = X.copy()

        transformed["temperature_difference"] = (
            transformed["process_temperature"] - transformed["air_temperature"]
        )

        angular_velocity = 2.0 * np.pi * transformed["rotational_speed"] / 60.0

        transformed["mechanical_power"] = transformed["torque"] * angular_velocity

        transformed["wear_torque_interaction"] = (
            transformed["tool_wear"] * transformed["torque"]
        )

        return transformed
