"""Data-drift detection utilities for MachineGuard."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset
from scipy.stats import chi2_contingency, ks_2samp


class DriftDetectionError(RuntimeError):
    """Raised when drift detection cannot be completed."""


@dataclass(frozen=True)
class ColumnDriftResult:
    """Drift result for one monitored column."""

    column: str
    column_type: str
    method: str
    score: float
    p_value: float | None
    effect_size: float
    threshold: float
    effect_size_threshold: float
    drift_detected: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class DriftResult:
    """Summary of one data-drift detection run."""

    generated_at: str
    reference_rows: int
    current_rows: int
    monitored_columns: list[str]
    drifted_columns: list[str]
    number_of_columns: int
    number_of_drifted_columns: int
    share_of_drifted_columns: float
    drift_share_threshold: float
    dataset_drift_detected: bool
    json_report_path: str
    html_report_path: str
    column_results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result."""

        return asdict(self)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load the drift-monitoring YAML configuration."""

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Drift configuration file was not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if "drift" not in config:
        raise DriftDetectionError(
            "The configuration must contain a 'drift' section."
        )

    if "paths" not in config:
        raise DriftDetectionError(
            "The configuration must contain a 'paths' section."
        )

    return config


def load_dataset(
    path: str | Path,
    dataset_name: str,
) -> pd.DataFrame:
    """Load and validate a monitoring dataset."""

    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} dataset was not found: {dataset_path}"
        )

    if dataset_path.suffix.lower() != ".csv":
        raise DriftDetectionError(
            f"{dataset_name} must be a CSV file: {dataset_path}"
        )

    dataframe = pd.read_csv(dataset_path)

    if dataframe.empty:
        raise DriftDetectionError(
            f"{dataset_name} dataset is empty: {dataset_path}"
        )

    if dataframe.columns.duplicated().any():
        duplicated = dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()

        raise DriftDetectionError(
            f"{dataset_name} contains duplicate columns: {duplicated}"
        )

    return dataframe


def _is_numeric_pair(
    reference_series: pd.Series,
    current_series: pd.Series,
) -> bool:
    """Return whether both series should be treated as numeric."""

    if pd.api.types.is_numeric_dtype(reference_series):
        return True

    if pd.api.types.is_numeric_dtype(current_series):
        return True

    reference_numeric = pd.to_numeric(
        reference_series,
        errors="coerce",
    )
    current_numeric = pd.to_numeric(
        current_series,
        errors="coerce",
    )

    reference_ratio = float(reference_numeric.notna().mean())
    current_ratio = float(current_numeric.notna().mean())

    return reference_ratio >= 0.95 and current_ratio >= 0.95


def prepare_monitoring_data(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    excluded_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align datasets and normalize comparable feature columns."""

    excluded = set(excluded_columns)

    reference_columns = {
        column
        for column in reference_data.columns
        if column not in excluded
    }

    current_columns = {
        column
        for column in current_data.columns
        if column not in excluded
    }

    common_columns = sorted(
        reference_columns & current_columns
    )

    if not common_columns:
        raise DriftDetectionError(
            "Reference and current datasets have no common "
            "monitoring columns."
        )

    missing_from_current = sorted(
        reference_columns - current_columns
    )
    missing_from_reference = sorted(
        current_columns - reference_columns
    )

    if missing_from_current:
        print(
            "Columns excluded because they are missing from "
            f"current data: {missing_from_current}"
        )

    if missing_from_reference:
        print(
            "Columns excluded because they are missing from "
            f"reference data: {missing_from_reference}"
        )

    reference_aligned = reference_data[
        common_columns
    ].copy()

    current_aligned = current_data[
        common_columns
    ].copy()

    for column in common_columns:
        reference_series = reference_aligned[column]
        current_series = current_aligned[column]

        if _is_numeric_pair(
            reference_series,
            current_series,
        ):
            reference_aligned[column] = pd.to_numeric(
                reference_series,
                errors="coerce",
            ).astype(float)

            current_aligned[column] = pd.to_numeric(
                current_series,
                errors="coerce",
            ).astype(float)

        else:
            reference_aligned[column] = (
                reference_series
                .astype("string")
                .fillna("__MISSING__")
            )

            current_aligned[column] = (
                current_series
                .astype("string")
                .fillna("__MISSING__")
            )

    return reference_aligned, current_aligned


def _validate_threshold(
    value: float,
    name: str,
) -> None:
    """Validate that a threshold is between zero and one."""

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )


def _clean_numeric_series(
    series: pd.Series,
) -> np.ndarray:
    """Return finite numeric values from a series."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=float)

    return values[np.isfinite(values)]


def _numeric_drift(
    column: str,
    reference_series: pd.Series,
    current_series: pd.Series,
    significance_level: float,
    minimum_effect_size: float,
) -> ColumnDriftResult:
    """Detect numerical drift using the two-sample KS test."""

    reference_values = _clean_numeric_series(
        reference_series
    )
    current_values = _clean_numeric_series(
        current_series
    )

    if reference_values.size == 0:
        raise DriftDetectionError(
            f"Reference column '{column}' has no valid numeric values."
        )

    if current_values.size == 0:
        raise DriftDetectionError(
            f"Current column '{column}' has no valid numeric values."
        )

    test_result = ks_2samp(
        reference_values,
        current_values,
        alternative="two-sided",
        method="auto",
    )

    statistic = float(test_result.statistic)
    p_value = float(test_result.pvalue)

    drift_detected = bool(
        p_value <= significance_level
        and statistic >= minimum_effect_size
    )

    return ColumnDriftResult(
        column=column,
        column_type="numerical",
        method="kolmogorov_smirnov",
        score=statistic,
        p_value=p_value,
        effect_size=statistic,
        threshold=significance_level,
        effect_size_threshold=minimum_effect_size,
        drift_detected=drift_detected,
    )


def _categorical_probabilities(
    reference_series: pd.Series,
    current_series: pd.Series,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build aligned categorical probability distributions."""

    reference_values = (
        reference_series
        .astype("string")
        .fillna("__MISSING__")
    )

    current_values = (
        current_series
        .astype("string")
        .fillna("__MISSING__")
    )

    categories = sorted(
        set(reference_values.unique().tolist())
        | set(current_values.unique().tolist())
    )

    reference_counts = (
        reference_values
        .value_counts()
        .reindex(categories, fill_value=0)
        .to_numpy(dtype=float)
    )

    current_counts = (
        current_values
        .value_counts()
        .reindex(categories, fill_value=0)
        .to_numpy(dtype=float)
    )

    reference_probabilities = (
        reference_counts / reference_counts.sum()
    )

    current_probabilities = (
        current_counts / current_counts.sum()
    )

    return (
        reference_probabilities,
        current_probabilities,
        categories,
    )


def _categorical_drift(
    column: str,
    reference_series: pd.Series,
    current_series: pd.Series,
    significance_level: float,
    minimum_effect_size: float,
) -> ColumnDriftResult:
    """Detect categorical drift using chi-square and total variation."""

    reference_values = (
        reference_series
        .astype("string")
        .fillna("__MISSING__")
    )

    current_values = (
        current_series
        .astype("string")
        .fillna("__MISSING__")
    )

    categories = sorted(
        set(reference_values.unique().tolist())
        | set(current_values.unique().tolist())
    )

    reference_counts = (
        reference_values
        .value_counts()
        .reindex(categories, fill_value=0)
        .to_numpy(dtype=float)
    )

    current_counts = (
        current_values
        .value_counts()
        .reindex(categories, fill_value=0)
        .to_numpy(dtype=float)
    )

    non_empty_categories = (
        reference_counts + current_counts
    ) > 0

    reference_counts = reference_counts[
        non_empty_categories
    ]
    current_counts = current_counts[
        non_empty_categories
    ]

    if len(reference_counts) <= 1:
        identical = np.array_equal(
            reference_counts,
            current_counts,
        )

        p_value = 1.0 if identical else 0.0
    else:
        contingency_table = np.vstack(
            [
                reference_counts,
                current_counts,
            ]
        )

        _, p_value, _, _ = chi2_contingency(
            contingency_table,
            correction=False,
        )

        p_value = float(p_value)

    (
        reference_probabilities,
        current_probabilities,
        _,
    ) = _categorical_probabilities(
        reference_series,
        current_series,
    )

    total_variation_distance = float(
        0.5
        * np.abs(
            reference_probabilities
            - current_probabilities
        ).sum()
    )

    drift_detected = bool(
        p_value <= significance_level
        and total_variation_distance
        >= minimum_effect_size
    )

    return ColumnDriftResult(
        column=column,
        column_type="categorical",
        method="chi_square_and_total_variation",
        score=total_variation_distance,
        p_value=p_value,
        effect_size=total_variation_distance,
        threshold=significance_level,
        effect_size_threshold=minimum_effect_size,
        drift_detected=drift_detected,
    )


def calculate_column_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    significance_level: float,
    numeric_effect_size_threshold: float,
    categorical_effect_size_threshold: float,
) -> list[ColumnDriftResult]:
    """Calculate stable per-column drift decisions."""

    results: list[ColumnDriftResult] = []

    for column in reference_data.columns:
        reference_series = reference_data[column]
        current_series = current_data[column]

        if pd.api.types.is_numeric_dtype(
            reference_series
        ):
            result = _numeric_drift(
                column=column,
                reference_series=reference_series,
                current_series=current_series,
                significance_level=significance_level,
                minimum_effect_size=(
                    numeric_effect_size_threshold
                ),
            )

        else:
            result = _categorical_drift(
                column=column,
                reference_series=reference_series,
                current_series=current_series,
                significance_level=significance_level,
                minimum_effect_size=(
                    categorical_effect_size_threshold
                ),
            )

        results.append(result)

    return results


def _safe_float(value: float) -> float:
    """Return a finite JSON-compatible float."""

    numeric_value = float(value)

    if math.isnan(numeric_value):
        return 0.0

    if math.isinf(numeric_value):
        return 1.0

    return numeric_value


def run_drift_detection(
    reference_path: str | Path,
    current_path: str | Path,
    json_report_path: str | Path,
    html_report_path: str | Path,
    summary_path: str | Path,
    excluded_columns: list[str] | None = None,
    drift_share_threshold: float = 0.30,
    method: str | None = None,
    significance_level: float = 0.05,
    numeric_effect_size_threshold: float = 0.10,
    categorical_effect_size_threshold: float = 0.10,
) -> DriftResult:
    """Run visual reporting and stable statistical drift detection."""

    _validate_threshold(
        drift_share_threshold,
        "drift_share_threshold",
    )

    _validate_threshold(
        significance_level,
        "significance_level",
    )

    _validate_threshold(
        numeric_effect_size_threshold,
        "numeric_effect_size_threshold",
    )

    _validate_threshold(
        categorical_effect_size_threshold,
        "categorical_effect_size_threshold",
    )

    excluded_columns = excluded_columns or []

    reference_data = load_dataset(
        reference_path,
        dataset_name="Reference",
    )

    current_data = load_dataset(
        current_path,
        dataset_name="Current",
    )

    (
        reference_aligned,
        current_aligned,
    ) = prepare_monitoring_data(
        reference_data=reference_data,
        current_data=current_data,
        excluded_columns=excluded_columns,
    )

    preset_arguments: dict[str, Any] = {
        "columns": list(reference_aligned.columns),
        "drift_share": drift_share_threshold,
    }

    if method:
        preset_arguments["method"] = method

    report = Report(
        metrics=[
            DataDriftPreset(
                **preset_arguments,
            ),
        ]
    )

    snapshot = report.run(
        reference_data=reference_aligned,
        current_data=current_aligned,
    )

    json_output = Path(json_report_path)
    html_output = Path(html_report_path)
    summary_output = Path(summary_path)

    json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    html_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot.save_json(
        str(json_output)
    )

    snapshot.save_html(
        str(html_output)
    )

    column_drift_results = calculate_column_drift(
        reference_data=reference_aligned,
        current_data=current_aligned,
        significance_level=significance_level,
        numeric_effect_size_threshold=(
            numeric_effect_size_threshold
        ),
        categorical_effect_size_threshold=(
            categorical_effect_size_threshold
        ),
    )

    drifted_columns = sorted(
        result.column
        for result in column_drift_results
        if result.drift_detected
    )

    number_of_columns = len(
        column_drift_results
    )

    number_of_drifted_columns = len(
        drifted_columns
    )

    share_of_drifted_columns = (
        number_of_drifted_columns
        / number_of_columns
        if number_of_columns
        else 0.0
    )

    dataset_drift_detected = bool(
        share_of_drifted_columns
        >= drift_share_threshold
    )

    serialized_column_results = []

    for result in column_drift_results:
        result_dictionary = result.to_dict()

        result_dictionary["score"] = _safe_float(
            result_dictionary["score"]
        )

        result_dictionary["effect_size"] = _safe_float(
            result_dictionary["effect_size"]
        )

        if result_dictionary["p_value"] is not None:
            result_dictionary["p_value"] = _safe_float(
                result_dictionary["p_value"]
            )

        serialized_column_results.append(
            result_dictionary
        )

    drift_result = DriftResult(
        generated_at=datetime.now(
            timezone.utc
        ).isoformat(),
        reference_rows=len(reference_aligned),
        current_rows=len(current_aligned),
        monitored_columns=list(
            reference_aligned.columns
        ),
        drifted_columns=drifted_columns,
        number_of_columns=number_of_columns,
        number_of_drifted_columns=(
            number_of_drifted_columns
        ),
        share_of_drifted_columns=float(
            share_of_drifted_columns
        ),
        drift_share_threshold=(
            drift_share_threshold
        ),
        dataset_drift_detected=(
            dataset_drift_detected
        ),
        json_report_path=str(json_output),
        html_report_path=str(html_output),
        column_results=serialized_column_results,
    )

    with summary_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            drift_result.to_dict(),
            file,
            indent=2,
            allow_nan=False,
        )

    return drift_result