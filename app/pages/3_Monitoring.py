"""MachineGuard monitoring dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

DRIFT_SUMMARY_PATH = PROJECT_ROOT / "reports" / "drift" / "drift_summary.json"

DRIFT_REPORT_PATH = PROJECT_ROOT / "reports" / "drift" / "drift_report.html"

PREDICTION_LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"

REQUEST_TIMEOUT_SECONDS = 5

st.set_page_config(
    page_title="MachineGuard Monitoring",
    page_icon="📊",
    layout="wide",
)

st.title("📊 MachineGuard Monitoring")

st.caption(
    "Monitor API activity, prediction risk, model versions and production data drift."
)


def load_json_file(
    path: Path,
) -> dict[str, Any]:
    """Load a JSON object safely from disk."""
    if not path.exists():
        return {}

    try:
        content = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}

    if isinstance(content, dict):
        return content

    return {}


@st.cache_data(ttl=10)
def load_drift_summary() -> dict[str, Any]:
    """Load the latest drift summary from API or disk."""
    try:
        response = requests.get(
            f"{API_URL}/monitoring/drift",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.ok:
            content = response.json()

            if isinstance(content, dict):
                return content

    except (
        requests.RequestException,
        ValueError,
    ):
        pass

    return load_json_file(DRIFT_SUMMARY_PATH)


@st.cache_data(ttl=10)
def load_api_status() -> dict[str, Any]:
    """Load API health and model readiness details."""
    status_data: dict[str, Any] = {
        "api_running": False,
        "health_status": "Unavailable",
        "model_loaded": False,
        "model_name": "Unavailable",
        "model_alias": "Unavailable",
        "model_version": "Unavailable",
    }

    try:
        health_response = requests.get(
            f"{API_URL}/health",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if health_response.ok:
            health_data = health_response.json()

            status_data["api_running"] = True
            status_data["health_status"] = health_data.get(
                "status",
                "healthy",
            )

        readiness_response = requests.get(
            f"{API_URL}/ready",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if readiness_response.ok:
            readiness_data = readiness_response.json()

            status_data["model_loaded"] = bool(
                readiness_data.get(
                    "model_loaded",
                    False,
                )
            )

            status_data["model_name"] = str(
                readiness_data.get(
                    "model_name",
                    "Unavailable",
                )
            )

            status_data["model_alias"] = str(
                readiness_data.get(
                    "model_alias",
                    "Unavailable",
                )
            )

            status_data["model_version"] = str(
                readiness_data.get(
                    "model_version",
                    "Unavailable",
                )
            )

    except (
        requests.RequestException,
        ValueError,
    ):
        pass

    return status_data


@st.cache_data(ttl=10)
def load_prediction_logs() -> pd.DataFrame:
    """Load prediction records from the JSON Lines log."""
    if not PREDICTION_LOG_PATH.exists():
        return pd.DataFrame()

    records: list[dict[str, Any]] = []

    try:
        with PREDICTION_LOG_PATH.open(
            mode="r",
            encoding="utf-8",
        ) as log_file:
            for line in log_file:
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)

                except json.JSONDecodeError:
                    continue

                if isinstance(record, dict):
                    records.append(record)

    except OSError:
        return pd.DataFrame()

    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        return dataframe

    if "timestamp" in dataframe.columns:
        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            errors="coerce",
            utc=True,
        )

        dataframe = dataframe.dropna(subset=["timestamp"])

    numeric_columns = [
        "prediction",
        "failure_probability",
        "latency_ms",
        "threshold",
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
    ]

    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    return dataframe


def get_successful_predictions(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Return only successful prediction records."""
    if dataframe.empty:
        return dataframe

    if "status" not in dataframe.columns:
        return dataframe

    return dataframe[dataframe["status"].astype(str).str.lower().eq("success")].copy()


def calculate_error_rate(
    dataframe: pd.DataFrame,
) -> float:
    """Calculate percentage of failed requests."""
    if dataframe.empty or "status" not in dataframe.columns:
        return 0.0

    error_count = dataframe["status"].astype(str).str.lower().eq("error").sum()

    return float(error_count) / len(dataframe) * 100


def format_drift_share(
    value: Any,
) -> str:
    """Format drift share as a percentage."""
    try:
        return f"{float(value):.1%}"

    except (
        TypeError,
        ValueError,
    ):
        return "Unavailable"


api_status = load_api_status()
drift_summary = load_drift_summary()
prediction_logs = load_prediction_logs()

successful_predictions = get_successful_predictions(prediction_logs)

total_requests = len(prediction_logs)
successful_count = len(successful_predictions)

failure_predictions = 0
average_probability = 0.0
average_latency = 0.0

if not successful_predictions.empty and "prediction" in successful_predictions.columns:
    failure_predictions = int(
        successful_predictions["prediction"].fillna(0).eq(1).sum()
    )

if (
    not successful_predictions.empty
    and "failure_probability" in successful_predictions.columns
):
    probability_values = successful_predictions["failure_probability"].dropna()

    if not probability_values.empty:
        average_probability = float(probability_values.mean())

if not prediction_logs.empty and "latency_ms" in prediction_logs.columns:
    latency_values = prediction_logs["latency_ms"].dropna()

    if not latency_values.empty:
        average_latency = float(latency_values.mean())

error_rate = calculate_error_rate(prediction_logs)

dataset_drift = drift_summary.get("dataset_drift")

drifted_columns_count = drift_summary.get("drifted_columns_count")

drifted_columns_share = drift_summary.get("drifted_columns_share")

st.subheader("Service status")

service_col_1, service_col_2, service_col_3 = st.columns(3)

with service_col_1:
    if api_status["api_running"]:
        st.success("API status: Running")
    else:
        st.error("API status: Unavailable")

with service_col_2:
    if api_status["model_loaded"]:
        st.success("Champion model: Loaded")
    else:
        st.warning("Champion model: Unavailable")

with service_col_3:
    st.info(f"Model version: {api_status['model_version']}")

st.caption(f"Model: {api_status['model_name']} | Alias: {api_status['model_alias']}")

st.divider()

st.subheader("Prediction monitoring")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Total requests",
    total_requests,
)

metric_2.metric(
    "Successful predictions",
    successful_count,
)

metric_3.metric(
    "Predicted failures",
    failure_predictions,
)

metric_4.metric(
    "Error rate",
    f"{error_rate:.2f}%",
)

metric_5, metric_6 = st.columns(2)

metric_5.metric(
    "Average failure probability",
    f"{average_probability:.2%}",
)

metric_6.metric(
    "Average latency",
    f"{average_latency:.2f} ms",
)

st.divider()

st.subheader("Production data drift")

drift_col_1, drift_col_2, drift_col_3 = st.columns(3)

with drift_col_1:
    if dataset_drift is True:
        st.error("Dataset drift detected")

    elif dataset_drift is False:
        st.success("No dataset-level drift")

    else:
        st.warning("Drift status unavailable")

with drift_col_2:
    st.metric(
        "Drifted columns",
        (drifted_columns_count if drifted_columns_count is not None else "Unavailable"),
    )

with drift_col_3:
    st.metric(
        "Drifted column share",
        format_drift_share(drifted_columns_share),
    )

summary_col, download_col = st.columns(2)

with summary_col:
    st.markdown("#### Latest report details")

    st.json(
        {
            "generated_at": drift_summary.get("generated_at"),
            "reference_rows": drift_summary.get("reference_rows"),
            "current_rows": drift_summary.get("current_rows"),
            "drifted_columns_count": (drifted_columns_count),
            "drifted_columns_share": (drifted_columns_share),
            "monitored_columns": (
                drift_summary.get(
                    "monitored_columns",
                    [],
                )
            ),
        }
    )

with download_col:
    st.markdown("#### Evidently report")

    report_path_value = drift_summary.get("html_report_path")

    if report_path_value:
        candidate_path = Path(str(report_path_value))

        if not candidate_path.is_absolute():
            candidate_path = PROJECT_ROOT / candidate_path

    else:
        candidate_path = DRIFT_REPORT_PATH

    if candidate_path.exists():
        st.download_button(
            label="Download HTML drift report",
            data=candidate_path.read_bytes(),
            file_name=("machineguard_drift_report.html"),
            mime="text/html",
        )

    else:
        st.info("Run `python monitoring/drift.py` to generate the HTML report.")

st.divider()

st.subheader("Recent prediction activity")

if prediction_logs.empty:
    st.info(
        "No prediction logs are available. "
        "Submit requests through the API or "
        "Streamlit prediction page."
    )

else:
    available_display_columns = [
        column
        for column in [
            "timestamp",
            "status",
            "machine_type",
            "prediction",
            "failure_probability",
            "risk_level",
            "latency_ms",
            "model_version",
            "prediction_id",
        ]
        if column in prediction_logs.columns
    ]

    recent_predictions = prediction_logs.sort_values(
        "timestamp",
        ascending=False,
    ).head(100)

    st.dataframe(
        recent_predictions[available_display_columns],
        use_container_width=True,
        hide_index=True,
    )

    if (
        "timestamp" in successful_predictions.columns
        and "failure_probability" in successful_predictions.columns
    ):
        chart_data = (
            successful_predictions.dropna(
                subset=[
                    "timestamp",
                    "failure_probability",
                ]
            )
            .sort_values("timestamp")
            .set_index("timestamp")
        )

        if not chart_data.empty:
            st.markdown("#### Failure probability over time")

            st.line_chart(chart_data[["failure_probability"]])

    if (
        "risk_level" in successful_predictions.columns
        and not successful_predictions.empty
    ):
        risk_distribution = (
            successful_predictions["risk_level"]
            .astype(str)
            .value_counts()
            .rename("predictions")
        )

        if not risk_distribution.empty:
            st.markdown("#### Risk-level distribution")

            st.bar_chart(risk_distribution)

st.caption(
    "The dashboard refreshes cached monitoring data approximately every 10 seconds."
)
