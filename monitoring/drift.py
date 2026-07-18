"""Generate MachineGuard data-drift reports using Evidently."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_PATH = PROJECT_ROOT / "data" / "reference" / "reference.csv"

CURRENT_PATH = PROJECT_ROOT / "data" / "current" / "current.csv"

REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "drift"

HTML_REPORT_PATH = REPORT_DIRECTORY / "drift_report.html"

JSON_REPORT_PATH = REPORT_DIRECTORY / "drift_report.json"

SUMMARY_PATH = REPORT_DIRECTORY / "drift_summary.json"

DATASET_DRIFT_THRESHOLD = 0.5

EXCLUDED_COLUMNS = {
    "target",
    "machine_failure",
    "failure_type",
    "product_id",
    "udi",
}


def load_dataset(
    path: Path,
) -> pd.DataFrame:
    """Load and validate a monitoring dataset.

    Args:
        path: CSV dataset path.

    Returns:
        Loaded dataset.

    Raises:
        FileNotFoundError: If the dataset does not exist.
        ValueError: If the dataset is empty.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset was not found: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {path}")

    return dataframe


def select_monitoring_columns(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select shared feature columns for drift monitoring.

    Args:
        reference: Training-time reference data.
        current: Current production-like data.

    Returns:
        Reference and current dataframes containing
        only comparable monitoring features.

    Raises:
        ValueError: If no common feature columns exist.
    """
    common_columns = [
        column
        for column in reference.columns
        if (column in current.columns and column.lower() not in EXCLUDED_COLUMNS)
    ]

    if not common_columns:
        raise ValueError(
            "Reference and current datasets have no common monitoring columns."
        )

    return (
        reference[common_columns].copy(),
        current[common_columns].copy(),
    )


def extract_drift_summary(
    report_dictionary: dict[str, Any],
) -> dict[str, Any]:
    """Extract dataset-level drift statistics.

    Evidently report structures may vary between
    versions, so report values are searched
    recursively.

    Args:
        report_dictionary: Evidently report dictionary.

    Returns:
        High-level drift summary.
    """
    summary: dict[str, Any] = {
        "generated_at": (datetime.now(UTC).isoformat()),
        "dataset_drift": None,
        "drifted_columns_count": None,
        "drifted_columns_share": None,
        "drifted_columns": [],
    }

    drifted_columns: list[str] = []

    def walk(
        value: Any,
    ) -> None:
        if isinstance(value, dict):
            column_name = (
                value.get("column_name") or value.get("column") or value.get("feature")
            )

            drift_detected = (
                value.get("drift_detected")
                if "drift_detected" in value
                else value.get("drifted")
            )

            if column_name is not None and drift_detected is True:
                column_value = str(column_name)

                if column_value not in drifted_columns:
                    drifted_columns.append(column_value)

            for key, nested_value in value.items():
                normalized_key = key.lower().replace("-", "_").replace(" ", "_")

                if normalized_key in {
                    "dataset_drift",
                    "datasetdrift",
                }:
                    if isinstance(
                        nested_value,
                        bool,
                    ):
                        summary["dataset_drift"] = nested_value

                elif normalized_key in {
                    "number_of_drifted_columns",
                    "drifted_columns_count",
                    "drifted_feature_count",
                }:
                    if isinstance(
                        nested_value,
                        (int, float),
                    ):
                        summary["drifted_columns_count"] = int(nested_value)

                elif normalized_key in {
                    "share_of_drifted_columns",
                    "drifted_columns_share",
                    "drifted_feature_share",
                }:
                    if isinstance(
                        nested_value,
                        (int, float),
                    ):
                        summary["drifted_columns_share"] = float(nested_value)

                walk(nested_value)

        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(report_dictionary)

    summary["drifted_columns"] = drifted_columns

    return summary


def generate_drift_report() -> dict[str, Any]:
    """Generate HTML, JSON and summary drift reports.

    Returns:
        High-level drift report metadata.
    """
    reference = load_dataset(REFERENCE_PATH)

    current = load_dataset(CURRENT_PATH)

    (
        reference_features,
        current_features,
    ) = select_monitoring_columns(
        reference=reference,
        current=current,
    )

    report = Report(
        [
            DataDriftPreset(
                drift_share=(DATASET_DRIFT_THRESHOLD),
            )
        ]
    )

    evaluation = report.run(
        current_data=current_features,
        reference_data=reference_features,
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation.save_html(str(HTML_REPORT_PATH))

    evaluation.save_json(str(JSON_REPORT_PATH))

    report_dictionary = evaluation.dict()

    summary = extract_drift_summary(report_dictionary)

    drifted_count = summary.get("drifted_columns_count")

    drifted_share = summary.get("drifted_columns_share")

    if drifted_count is None and summary["drifted_columns"]:
        drifted_count = len(summary["drifted_columns"])

        summary["drifted_columns_count"] = drifted_count

    if (
        drifted_share is None
        and drifted_count is not None
        and len(reference_features.columns) > 0
    ):
        drifted_share = float(drifted_count) / len(reference_features.columns)

        summary["drifted_columns_share"] = drifted_share

    if drifted_share is not None:
        summary["dataset_drift"] = float(drifted_share) >= DATASET_DRIFT_THRESHOLD

    summary.update(
        {
            "drift_threshold": (DATASET_DRIFT_THRESHOLD),
            "reference_rows": len(reference_features),
            "current_rows": len(current_features),
            "monitored_columns": list(reference_features.columns),
            "html_report_path": str(HTML_REPORT_PATH.relative_to(PROJECT_ROOT)),
            "json_report_path": str(JSON_REPORT_PATH.relative_to(PROJECT_ROOT)),
        }
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    """Run drift detection from the command line."""
    summary = generate_drift_report()

    print("Drift report generated successfully.")

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
