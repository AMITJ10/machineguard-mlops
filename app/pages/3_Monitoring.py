from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"

DRIFT_SUMMARY_PATH = PROJECT_ROOT / "reports" / "drift" / "drift_summary.json"

DRIFT_REPORT_PATH = PROJECT_ROOT / "reports" / "drift" / "drift_report.html"

RETRAINING_STATUS_PATH = PROJECT_ROOT / "reports" / "retraining" / "status.json"

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

FEATURE_COLUMNS = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]

EXPECTED_INPUT_COLUMNS = [
    "machine_type",
    *FEATURE_COLUMNS,
]

REQUEST_TIMEOUT_SECONDS = 5


st.set_page_config(
    page_title="MachineGuard Monitoring",
    page_icon="📊",
    layout="wide",
)


def load_json_file(
    path: Path,
) -> dict[str, Any]:
    """Read a JSON object from disk."""
    if not path.exists():
        return {}

    try:
        with path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            content = json.load(file)

        if isinstance(content, dict):
            return content

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}

    return {}


@st.cache_data(ttl=15)
def load_prediction_logs() -> pd.DataFrame:
    """Load prediction events from the JSON Lines log."""
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
            utc=True,
            errors="coerce",
        )

        dataframe = dataframe.dropna(subset=["timestamp"])

        dataframe = dataframe.sort_values("timestamp")

    numeric_columns = [
        "latency_ms",
        "prediction",
        "failure_probability",
        "threshold",
        *FEATURE_COLUMNS,
    ]

    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    return dataframe


@st.cache_data(ttl=10)
def get_api_status() -> dict[str, Any]:
    """Read API health and readiness information."""
    result: dict[str, Any] = {
        "api_running": False,
        "health_status": "unavailable",
        "ready_status": "unavailable",
        "model_loaded": False,
        "model_name": "Unknown",
        "model_alias": "Unknown",
        "model_version": "Unknown",
        "error": None,
    }

    try:
        health_response = requests.get(
            f"{API_URL}/health",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if health_response.ok:
            health_data = health_response.json()

            result["api_running"] = True
            result["health_status"] = health_data.get(
                "status",
                "healthy",
            )

        ready_response = requests.get(
            f"{API_URL}/ready",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if ready_response.ok:
            ready_data = ready_response.json()

            result["ready_status"] = ready_data.get(
                "status",
                "ready",
            )

            result["model_loaded"] = bool(
                ready_data.get(
                    "model_loaded",
                    False,
                )
            )

            result["model_name"] = ready_data.get(
                "model_name",
                "Unknown",
            )

            result["model_alias"] = ready_data.get(
                "model_alias",
                "Unknown",
            )

            result["model_version"] = str(
                ready_data.get(
                    "model_version",
                    "Unknown",
                )
            )

    except requests.RequestException as error:
        result["error"] = str(error)

    return result


@st.cache_data(ttl=10)
def get_process_memory_mb() -> float | None:
    """Read API process memory from Prometheus metrics."""
    try:
        response = requests.get(
            f"{API_URL}/metrics",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

    except requests.RequestException:
        return None

    for line in response.text.splitlines():
        if line.startswith("process_resident_memory_bytes "):
            try:
                memory_bytes = float(line.split()[-1])

                return memory_bytes / (1024 * 1024)

            except (
                ValueError,
                IndexError,
            ):
                return None

    return None


def calculate_error_rate(
    dataframe: pd.DataFrame,
) -> float:
    """Calculate percentage of logged failed requests."""
    if dataframe.empty or "status" not in dataframe.columns:
        return 0.0

    error_count = dataframe["status"].astype(str).str.lower().eq("error").sum()

    return float(error_count) / len(dataframe) * 100


def calculate_missing_rate(
    dataframe: pd.DataFrame,
) -> float:
    """Calculate overall missing-input percentage."""
    available_columns = [
        column for column in EXPECTED_INPUT_COLUMNS if column in dataframe.columns
    ]

    if dataframe.empty or not available_columns:
        return 0.0

    missing_values = dataframe[available_columns].isna().sum().sum()

    total_values = len(dataframe) * len(available_columns)

    if total_values == 0:
        return 0.0

    return float(missing_values) / total_values * 100


def calculate_invalid_request_rate(
    dataframe: pd.DataFrame,
) -> float:
    """Calculate requests rejected or logged as errors."""
    if dataframe.empty or "status" not in dataframe.columns:
        return 0.0

    invalid_count = (
        dataframe["status"]
        .astype(str)
        .str.lower()
        .isin(
            {
                "error",
                "invalid",
                "validation_error",
            }
        )
        .sum()
    )

    return float(invalid_count) / len(dataframe) * 100


def extract_drifted_features(
    summary: dict[str, Any],
) -> list[str]:
    """Extract drifted features from different report formats."""
    candidates = [
        summary.get("drifted_features"),
        summary.get("drifted_columns"),
        summary.get("features_with_drift"),
    ]

    for candidate in candidates:
        if isinstance(candidate, list):
            return [str(feature) for feature in candidate]

    feature_results = summary.get("feature_results")

    if isinstance(feature_results, dict):
        return [
            str(feature)
            for feature, result in feature_results.items()
            if isinstance(result, dict)
            and bool(
                result.get(
                    "drift_detected",
                    result.get(
                        "drifted",
                        False,
                    ),
                )
            )
        ]

    count = summary.get("drifted_feature_count")

    if isinstance(count, int):
        return [f"Feature {index + 1}" for index in range(count)]

    return []


def latest_drift_report_date(
    summary: dict[str, Any],
) -> str:
    """Return latest drift report generation date."""
    date_keys = [
        "generated_at",
        "report_date",
        "created_at",
        "timestamp",
    ]

    for key in date_keys:
        value = summary.get(key)

        if value:
            return str(value)

    if DRIFT_SUMMARY_PATH.exists():
        modified_time = datetime.fromtimestamp(
            DRIFT_SUMMARY_PATH.stat().st_mtime,
            tz=timezone.utc,
        )

        return modified_time.isoformat()

    return "Not generated"


def get_retraining_status() -> dict[str, str]:
    """Read retraining status if a status file exists."""
    status_data = load_json_file(RETRAINING_STATUS_PATH)

    return {
        "status": str(
            status_data.get(
                "status",
                "Not configured",
            )
        ),
        "last_run": str(
            status_data.get(
                "last_run",
                "Never",
            )
        ),
        "message": str(
            status_data.get(
                "message",
                "No retraining workflow has reported a status.",
            )
        ),
    }


def create_requests_over_time(
    dataframe: pd.DataFrame,
) -> None:
    """Render request volume over time."""
    if dataframe.empty or "timestamp" not in dataframe.columns:
        st.info("No timestamped prediction requests are available.")
        return

    requests = (
        dataframe.set_index("timestamp").resample("5min").size().rename("requests")
    )

    st.line_chart(requests)


def create_latency_chart(
    dataframe: pd.DataFrame,
) -> None:
    """Render average latency over time."""
    if (
        dataframe.empty
        or "latency_ms" not in dataframe.columns
        or "timestamp" not in dataframe.columns
    ):
        st.info("Latency data is not available.")
        return

    latency = (
        dataframe.set_index("timestamp")["latency_ms"]
        .resample("5min")
        .mean()
        .dropna()
        .rename("average_latency_ms")
    )

    st.line_chart(latency)


def create_risk_distribution(
    dataframe: pd.DataFrame,
) -> None:
    """Render prediction risk distribution."""
    if dataframe.empty or "risk_level" not in dataframe.columns:
        st.info("Risk-level data is not available.")
        return

    risk_order = [
        "low",
        "medium",
        "high",
        "critical",
    ]

    distribution = (
        dataframe["risk_level"]
        .astype(str)
        .str.lower()
        .value_counts()
        .reindex(
            risk_order,
            fill_value=0,
        )
        .rename("predictions")
    )

    st.bar_chart(distribution)


def create_failure_rate_chart(
    dataframe: pd.DataFrame,
) -> None:
    """Render predicted failure rate over time."""
    if (
        dataframe.empty
        or "prediction" not in dataframe.columns
        or "timestamp" not in dataframe.columns
    ):
        st.info("Prediction outcome data is not available.")
        return

    successful = dataframe.copy()

    if "status" in successful.columns:
        successful = successful[
            successful["status"].astype(str).str.lower().eq("success")
        ]

    failure_rate = (
        successful.set_index("timestamp")["prediction"]
        .resample("30min")
        .mean()
        .mul(100)
        .dropna()
        .rename("predicted_failure_rate_percent")
    )

    st.line_chart(failure_rate)


def create_probability_chart(
    dataframe: pd.DataFrame,
) -> None:
    """Render predicted failure-probability distribution."""
    if dataframe.empty or "failure_probability" not in dataframe.columns:
        st.info("Probability data is not available.")
        return

    probabilities = dataframe[["failure_probability"]].dropna()

    if probabilities.empty:
        st.info("Probability data is not available.")
        return

    histogram = pd.cut(
        probabilities["failure_probability"],
        bins=[
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ],
        include_lowest=True,
    ).value_counts(sort=False)

    histogram.index = histogram.index.astype(str)

    st.bar_chart(histogram.rename("predictions"))


def render_feature_distributions(
    dataframe: pd.DataFrame,
) -> None:
    """Render feature distributions."""
    available_features = [
        feature for feature in FEATURE_COLUMNS if feature in dataframe.columns
    ]

    if not available_features:
        st.info("No numeric input features are available.")
        return

    selected_feature = st.selectbox(
        "Select feature",
        options=available_features,
    )

    feature_values = dataframe[selected_feature].dropna()

    if feature_values.empty:
        st.info(f"No values are available for {selected_feature}.")
        return

    bins = min(
        20,
        max(
            5,
            int(feature_values.nunique()),
        ),
    )

    histogram = pd.cut(
        feature_values,
        bins=bins,
        duplicates="drop",
    ).value_counts(sort=False)

    histogram.index = histogram.index.astype(str)

    st.bar_chart(histogram.rename("records"))

    st.dataframe(
        feature_values.describe().to_frame(name="value"),
        use_container_width=True,
    )


def render_missing_rates(
    dataframe: pd.DataFrame,
) -> None:
    """Render missing input rates per feature."""
    available_columns = [
        column for column in EXPECTED_INPUT_COLUMNS if column in dataframe.columns
    ]

    if not available_columns:
        st.info("Input columns are not available.")
        return

    missing_rates = (
        dataframe[available_columns]
        .isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .rename("missing_rate_percent")
    )

    st.bar_chart(missing_rates)

    st.dataframe(
        missing_rates.to_frame(),
        use_container_width=True,
    )


def render_drift_details(
    drift_summary: dict[str, Any],
) -> None:
    """Render drift summary details."""
    drifted_features = extract_drifted_features(drift_summary)

    if drifted_features:
        st.warning("Drift detected in: " + ", ".join(drifted_features))
    else:
        st.success("No drifted features were reported.")

    if drift_summary:
        st.json(drift_summary)
    else:
        st.info("Run `python monitoring/drift.py` to generate the drift summary.")

    if DRIFT_REPORT_PATH.exists():
        try:
            report_html = DRIFT_REPORT_PATH.read_text(encoding="utf-8")

            st.download_button(
                label="Download drift report",
                data=report_html,
                file_name="drift_report.html",
                mime="text/html",
            )

        except OSError:
            st.warning("The drift report could not be read.")


st.title("📊 MachineGuard Monitoring Dashboard")

st.caption(
    "Service, data, drift and model monitoring for the MachineGuard prediction system."
)

if st.button(
    "Refresh monitoring data",
    type="primary",
):
    st.cache_data.clear()
    st.rerun()

prediction_logs = load_prediction_logs()
api_status = get_api_status()
process_memory_mb = get_process_memory_mb()
drift_summary = load_json_file(DRIFT_SUMMARY_PATH)
retraining_status = get_retraining_status()

total_predictions = len(prediction_logs)

average_latency = 0.0

if not prediction_logs.empty and "latency_ms" in prediction_logs.columns:
    average_latency = float(prediction_logs["latency_ms"].mean())

error_rate = calculate_error_rate(prediction_logs)

missing_rate = calculate_missing_rate(prediction_logs)

invalid_request_rate = calculate_invalid_request_rate(prediction_logs)

drifted_features = extract_drifted_features(drift_summary)

st.subheader("System summary")

column_1, column_2, column_3, column_4 = st.columns(4)

column_1.metric(
    "Total predictions",
    f"{total_predictions:,}",
)

column_2.metric(
    "Average API latency",
    f"{average_latency:.2f} ms",
)

column_3.metric(
    "HTTP/error rate",
    f"{error_rate:.2f}%",
)

column_4.metric(
    "Drifted feature count",
    len(drifted_features),
)

column_5, column_6, column_7, column_8 = st.columns(4)

column_5.metric(
    "Missing input rate",
    f"{missing_rate:.2f}%",
)

column_6.metric(
    "Invalid request rate",
    f"{invalid_request_rate:.2f}%",
)

column_7.metric(
    "Active model version",
    api_status["model_version"],
)

column_8.metric(
    "Retraining status",
    retraining_status["status"],
)

service_tab, data_tab, model_tab, drift_tab = st.tabs(
    [
        "Service monitoring",
        "Data monitoring",
        "Model monitoring",
        "Drift monitoring",
    ]
)

with service_tab:
    st.subheader("API status")

    status_column_1, status_column_2, status_column_3 = st.columns(3)

    if api_status["api_running"]:
        status_column_1.success("API is running")
    else:
        status_column_1.error("API is unavailable")

    if api_status["model_loaded"]:
        status_column_2.success("Model is loaded")
    else:
        status_column_2.error("Model is not loaded")

    if process_memory_mb is None:
        status_column_3.metric(
            "API memory usage",
            "Unavailable",
        )
    else:
        status_column_3.metric(
            "API memory usage",
            f"{process_memory_mb:.2f} MB",
        )

    st.write(
        {
            "API URL": API_URL,
            "Health status": (api_status["health_status"]),
            "Readiness status": (api_status["ready_status"]),
        }
    )

    if api_status["error"]:
        st.error(api_status["error"])

    chart_column_1, chart_column_2 = st.columns(2)

    with chart_column_1:
        st.markdown("#### Requests over time")
        create_requests_over_time(prediction_logs)

    with chart_column_2:
        st.markdown("#### Average latency over time")
        create_latency_chart(prediction_logs)

    if average_latency > 500:
        st.warning("Average API latency is above 500 ms.")
    elif total_predictions > 0:
        st.success("Average API latency is within the current alert threshold.")

    if error_rate > 5:
        st.error("The request error rate is above 5%.")

with data_tab:
    st.subheader("Input data quality")

    data_column_1, data_column_2 = st.columns(2)

    with data_column_1:
        st.markdown("#### Feature distributions")
        render_feature_distributions(prediction_logs)

    with data_column_2:
        st.markdown("#### Missing input rates")
        render_missing_rates(prediction_logs)

    if "machine_type" in prediction_logs.columns:
        st.markdown("#### Machine-type distribution")

        category_distribution = (
            prediction_logs["machine_type"]
            .fillna("missing")
            .astype(str)
            .value_counts()
            .rename("records")
        )

        st.bar_chart(category_distribution)

        unexpected_categories = sorted(
            set(prediction_logs["machine_type"].dropna().astype(str)) - {"L", "M", "H"}
        )

        if unexpected_categories:
            st.error(
                "Unexpected categories detected: " + ", ".join(unexpected_categories)
            )

with model_tab:
    st.subheader("Model behaviour")

    model_column_1, model_column_2 = st.columns(2)

    with model_column_1:
        st.markdown("#### Failure-risk distribution")
        create_risk_distribution(prediction_logs)

    with model_column_2:
        st.markdown("#### Probability distribution")
        create_probability_chart(prediction_logs)

    st.markdown("#### Predicted failure rate over time")

    create_failure_rate_chart(prediction_logs)

    st.write(
        {
            "Model name": (api_status["model_name"]),
            "Model alias": (api_status["model_alias"]),
            "Model version": (api_status["model_version"]),
            "Retraining status": (retraining_status["status"]),
            "Last retraining run": (retraining_status["last_run"]),
            "Retraining message": (retraining_status["message"]),
        }
    )

    st.info(
        "Recall, precision and calibration require actual "
        "failure labels. Prediction logs alone cannot measure "
        "true model performance."
    )

with drift_tab:
    st.subheader("Feature drift")

    drift_column_1, drift_column_2 = st.columns(2)

    drift_column_1.metric(
        "Drifted features",
        len(drifted_features),
    )

    drift_column_2.metric(
        "Latest drift report",
        latest_drift_report_date(drift_summary),
    )

    render_drift_details(drift_summary)

if prediction_logs.empty:
    st.warning(
        "No prediction records were found. Submit predictions "
        "through the API before using the dashboard."
    )
else:
    with st.expander("View recent prediction events"):
        st.dataframe(
            prediction_logs.tail(100),
            use_container_width=True,
            hide_index=True,
        )
