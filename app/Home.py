from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

REQUEST_TIMEOUT_SECONDS = 30


def get_api_readiness() -> dict[str, Any] | None:
    """Return API readiness information when available."""
    try:
        response = requests.get(
            f"{API_URL}/ready",
            timeout=5,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def submit_prediction(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Send one machine record to the prediction API."""
    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    return response.json()


def display_prediction(
    result: dict[str, Any],
) -> None:
    """Display prediction information."""
    probability = float(result["failure_probability"])

    risk_level = str(result["risk_level"]).lower()

    prediction = int(result["prediction"])

    left, middle, right = st.columns(3)

    with left:
        st.metric(
            label="Failure probability",
            value=f"{probability:.2%}",
        )

    with middle:
        st.metric(
            label="Predicted class",
            value=("Failure" if prediction == 1 else "No failure"),
        )

    with right:
        st.metric(
            label="Risk level",
            value=risk_level.title(),
        )

    if risk_level == "critical":
        st.error(
            "Critical failure risk. Stop the machine safely "
            "and arrange an immediate technical inspection."
        )

    elif risk_level == "high":
        st.error("High failure risk. Immediate inspection is recommended.")

    elif risk_level == "medium":
        st.warning("Medium failure risk. Schedule a maintenance inspection soon.")

    else:
        st.success("Low failure risk under the current readings.")

    with st.expander(
        "Prediction details",
        expanded=False,
    ):
        st.json(result)


st.set_page_config(
    page_title="MachineGuard AI",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ MachineGuard AI")

st.write("Predict industrial machine failure risk from live machine measurements.")

readiness = get_api_readiness()

if readiness is None:
    st.error(
        "The MachineGuard API is not available. Start FastAPI "
        "before submitting a prediction."
    )

elif readiness.get("status") == "ready":
    st.success(
        "Prediction service ready · "
        f"Model {readiness.get('model_name')} · "
        f"Version {readiness.get('model_version')} · "
        f"Alias {readiness.get('model_alias')}"
    )

else:
    st.warning("The API is running, but the production model is not ready.")

st.divider()

with st.form(
    key="prediction_form",
    clear_on_submit=False,
):
    st.subheader("Machine measurements")

    left, right = st.columns(2)

    with left:
        machine_type = st.selectbox(
            label="Machine type",
            options=["L", "M", "H"],
            index=1,
            help=(
                "L = low-quality machine, "
                "M = medium-quality machine, "
                "H = high-quality machine."
            ),
        )

        air_temperature = st.number_input(
            label="Air temperature (K)",
            min_value=250.0,
            max_value=400.0,
            value=298.1,
            step=0.1,
            format="%.1f",
        )

        process_temperature = st.number_input(
            label="Process temperature (K)",
            min_value=250.0,
            max_value=450.0,
            value=308.6,
            step=0.1,
            format="%.1f",
        )

    with right:
        rotational_speed = st.number_input(
            label="Rotational speed (RPM)",
            min_value=0.0,
            max_value=5000.0,
            value=1551.0,
            step=1.0,
            format="%.1f",
        )

        torque = st.number_input(
            label="Torque (Nm)",
            min_value=0.0,
            max_value=200.0,
            value=42.8,
            step=0.1,
            format="%.1f",
        )

        tool_wear = st.number_input(
            label="Tool wear (minutes)",
            min_value=0.0,
            max_value=500.0,
            value=0.0,
            step=1.0,
            format="%.1f",
        )

    submitted = st.form_submit_button(
        label="Predict failure risk",
        type="primary",
        use_container_width=True,
        disabled=readiness is None,
    )

if submitted:
    request_payload = {
        # The public API expects machine_type.
        # prediction_service.py converts it internally to type.
        "machine_type": machine_type,
        "air_temperature": float(air_temperature),
        "process_temperature": float(process_temperature),
        "rotational_speed": float(rotational_speed),
        "torque": float(torque),
        "tool_wear": float(tool_wear),
    }

    try:
        with st.spinner("Calculating failure risk..."):
            prediction_result = submit_prediction(request_payload)

        display_prediction(prediction_result)

    except requests.Timeout:
        st.error(
            "The prediction request timed out. Check whether "
            "FastAPI and MLflow are running correctly."
        )

    except requests.HTTPError as error:
        response_text = ""

        if error.response is not None:
            response_text = error.response.text

        st.error("The API rejected the prediction request.")

        if response_text:
            st.code(
                response_text,
                language="json",
            )

    except requests.RequestException as error:
        st.error(f"Could not connect to the prediction API: {error}")
