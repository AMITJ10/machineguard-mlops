from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


REFERENCE_PATH = Path("data/reference/reference.csv")

PREDICTION_LOG_PATH = Path("logs/predictions.jsonl")

CURRENT_PATH = Path("data/current/current.csv")

REPORT_PATH = Path("reports/drift/drift_report.html")

REPORT_JSON_PATH = Path("reports/drift/drift_report.json")

SUMMARY_PATH = Path("reports/drift/drift_summary.json")

FEATURE_COLUMNS = [
    "type",
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]

NUMERICAL_COLUMNS = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]

CATEGORICAL_COLUMNS = [
    "type",
]

PSI_DRIFT_THRESHOLD = 0.20


def _utc_timestamp() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def load_prediction_logs() -> pd.DataFrame:
    """Load successful prediction events from JSON Lines."""
    if not PREDICTION_LOG_PATH.exists():
        raise FileNotFoundError(f"Prediction log not found: {PREDICTION_LOG_PATH}")

    records: list[dict[str, Any]] = []

    with PREDICTION_LOG_PATH.open(
        mode="r",
        encoding="utf-8",
    ) as log_file:
        for line_number, line in enumerate(
            log_file,
            start=1,
        ):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_number}.")
                continue

            if record.get("status") == "success":
                records.append(record)

    if not records:
        raise ValueError("No successful prediction records were found.")

    return pd.DataFrame.from_records(records)


def create_current_dataset() -> pd.DataFrame:
    """Create current feature data from production logs."""
    logs = load_prediction_logs()

    logs = logs.rename(
        columns={
            "machine_type": "type",
        }
    )

    missing_columns = [
        column for column in FEATURE_COLUMNS if column not in logs.columns
    ]

    if missing_columns:
        raise ValueError(f"Prediction logs are missing columns: {missing_columns}")

    current = logs.loc[
        :,
        FEATURE_COLUMNS,
    ].copy()

    for column in NUMERICAL_COLUMNS:
        current[column] = pd.to_numeric(
            current[column],
            errors="coerce",
        )

    current["type"] = current["type"].astype("string").str.upper()

    current = current.dropna(subset=FEATURE_COLUMNS)

    if current.empty:
        raise ValueError("Current data became empty after cleaning.")

    CURRENT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    current.to_csv(
        CURRENT_PATH,
        index=False,
    )

    return current


def validate_reference_data(
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize the reference dataset."""
    missing_columns = [
        column for column in FEATURE_COLUMNS if column not in reference.columns
    ]

    if missing_columns:
        raise ValueError(f"Reference data is missing columns: {missing_columns}")

    reference = reference.loc[
        :,
        FEATURE_COLUMNS,
    ].copy()

    for column in NUMERICAL_COLUMNS:
        reference[column] = pd.to_numeric(
            reference[column],
            errors="coerce",
        )

    reference["type"] = reference["type"].astype("string").str.upper()

    reference = reference.dropna(subset=FEATURE_COLUMNS)

    if reference.empty:
        raise ValueError("Reference dataset is empty after cleaning.")

    return reference


def calculate_numeric_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Calculate Population Stability Index for numeric data."""
    reference_values = (
        pd.to_numeric(
            reference,
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )

    current_values = (
        pd.to_numeric(
            current,
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )

    if reference_values.size == 0 or current_values.size == 0:
        return float("nan")

    quantiles = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    bin_edges = np.unique(
        np.quantile(
            reference_values,
            quantiles,
        )
    )

    if bin_edges.size < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference_values,
        bins=bin_edges,
    )

    current_counts, _ = np.histogram(
        current_values,
        bins=bin_edges,
    )

    epsilon = 1e-6

    reference_proportions = reference_counts / max(
        reference_counts.sum(),
        1,
    )

    current_proportions = current_counts / max(
        current_counts.sum(),
        1,
    )

    reference_proportions = np.clip(
        reference_proportions,
        epsilon,
        None,
    )

    current_proportions = np.clip(
        current_proportions,
        epsilon,
        None,
    )

    psi = np.sum(
        (current_proportions - reference_proportions)
        * np.log(current_proportions / reference_proportions)
    )

    return float(psi)


def calculate_categorical_psi(
    reference: pd.Series,
    current: pd.Series,
) -> float:
    """Calculate PSI for categorical data."""
    reference_values = reference.astype("string").fillna("__MISSING__")

    current_values = current.astype("string").fillna("__MISSING__")

    categories = sorted(set(reference_values.unique()) | set(current_values.unique()))

    reference_distribution = reference_values.value_counts(normalize=True).reindex(
        categories,
        fill_value=0.0,
    )

    current_distribution = current_values.value_counts(normalize=True).reindex(
        categories,
        fill_value=0.0,
    )

    epsilon = 1e-6

    reference_distribution = reference_distribution.clip(lower=epsilon)

    current_distribution = current_distribution.clip(lower=epsilon)

    psi = (
        (current_distribution - reference_distribution)
        * np.log(current_distribution / reference_distribution)
    ).sum()

    return float(psi)


def calculate_drift_summary(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate dashboard-friendly feature drift results."""
    feature_results: dict[
        str,
        dict[str, Any],
    ] = {}

    for column in NUMERICAL_COLUMNS:
        psi_score = calculate_numeric_psi(
            reference[column],
            current[column],
        )

        feature_results[column] = {
            "feature_type": "numerical",
            "psi": (None if np.isnan(psi_score) else round(psi_score, 6)),
            "drift_detected": bool(
                not np.isnan(psi_score) and psi_score >= PSI_DRIFT_THRESHOLD
            ),
        }

    for column in CATEGORICAL_COLUMNS:
        psi_score = calculate_categorical_psi(
            reference[column],
            current[column],
        )

        feature_results[column] = {
            "feature_type": "categorical",
            "psi": round(
                psi_score,
                6,
            ),
            "drift_detected": bool(psi_score >= PSI_DRIFT_THRESHOLD),
        }

    drifted_features = [
        feature
        for feature, result in feature_results.items()
        if result["drift_detected"]
    ]

    return {
        "generated_at": _utc_timestamp(),
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "psi_threshold": (PSI_DRIFT_THRESHOLD),
        "total_features": len(FEATURE_COLUMNS),
        "drifted_feature_count": len(drifted_features),
        "dataset_drift_detected": bool(drifted_features),
        "drifted_features": (drifted_features),
        "features": feature_results,
        "evidently_report": str(REPORT_PATH),
    }


def generate_drift_report() -> None:
    """Generate Evidently and dashboard drift reports."""
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Reference dataset not found: {REFERENCE_PATH}")

    reference = pd.read_csv(REFERENCE_PATH)

    reference = validate_reference_data(reference)

    current = create_current_dataset()

    report = Report(
        [
            DataDriftPreset(
                columns=FEATURE_COLUMNS,
            ),
        ]
    )

    snapshot = report.run(
        current_data=current,
        reference_data=reference,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot.save_html(str(REPORT_PATH))

    snapshot.save_json(str(REPORT_JSON_PATH))

    summary = calculate_drift_summary(
        reference=reference,
        current=current,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Current dataset saved to {CURRENT_PATH}")

    print(f"Evidently drift report saved to {REPORT_PATH}")

    print(f"Drift summary saved to {SUMMARY_PATH}")

    print(f"Drifted features: {summary['drifted_features']}")


if __name__ == "__main__":
    generate_drift_report()
