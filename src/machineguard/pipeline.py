from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.machineguard.features import MachineFeatureEngineer


NUMERIC_FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "temperature_difference",
    "mechanical_power",
    "wear_torque_interaction",
]

CATEGORICAL_FEATURES = [
    "type",
]


def build_training_pipeline(
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    random_state: int = 42,
) -> Pipeline:
    """Build the complete preprocessing and classification pipeline.

    Args:
        n_estimators: Number of trees in the random forest.
        max_depth: Maximum depth of each tree.
        min_samples_split: Minimum samples required to split a node.
        random_state: Seed used for reproducibility.

    Returns:
        A complete Scikit-learn training pipeline.
    """
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "feature_engineering",
                MachineFeatureEngineer(),
            ),
            (
                "preprocessing",
                preprocessing,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )
