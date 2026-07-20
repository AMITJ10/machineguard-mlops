"""MachineGuard AI Dashboard."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

st.set_page_config(
    page_title="MachineGuard AI",
    page_icon="⚙️",
    layout="wide",
)

API_URL = os.getenv(
    "API_URL",
    "https://machineguard-mlops.onrender.com",
).rstrip("/")

READINESS_TIMEOUT_SECONDS = 5
PREDICTION_TIMEOUT_SECONDS = 30


def get_api_readiness() -> dict[str, Any] | None:
    """Return API readiness information."""

    try:
        response = requests.get(
            f"{API_URL}/ready",
            timeout=READINESS_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        result = response.json()

        if isinstance(result, dict):
            return result

        return None

    except (
        requests.RequestException,
        ValueError,
    ):
        return None


def submit_prediction(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Send prediction request."""

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=PREDICTION_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    result = response.json()

    if not isinstance(result, dict):
        raise ValueError(
            "Prediction API returned an invalid response."
        )

    return result


def display_prediction(
    result: dict[str, Any],
) -> None:
    """Display prediction dashboard."""

    probability = float(result["failure_probability"])

    prediction = int(result["prediction"])

    risk = str(result["risk_level"]).upper()

    st.divider()

    st.subheader("Prediction Result")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Failure Probability",
            f"{probability:.2%}",
        )

    with c2:
        st.metric(
            "Prediction",
            "Failure"
            if prediction
            else "Healthy",
        )

    with c3:
        st.metric(
            "Risk",
            risk,
        )

    st.progress(probability)

    if risk == "LOW":
        st.success(
            "Machine operating normally."
        )

    elif risk == "MEDIUM":
        st.warning(
            "Maintenance should be scheduled."
        )

    elif risk == "HIGH":
        st.error(
            "High probability of failure."
        )

    else:
        st.error(
            "Critical risk detected."
        )

    with st.expander(
        "Raw API Response",
        expanded=False,
    ):
        st.json(result)


# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:

    st.title("⚙️ MachineGuard")

    st.markdown(
        """
Industrial Predictive Maintenance Platform

---

### Tech Stack

- FastAPI
- Scikit-Learn
- MLflow
- Docker
- Streamlit
- Render
- AWS (Next Phase)

---
"""
    )

    st.link_button(
        "API Docs",
        f"{API_URL}/docs",
    )

    st.link_button(
        "Health Check",
        f"{API_URL}/health",
    )

    st.link_button(
        "GitHub Repository",
        "https://github.com/AMITJ10/machineguard-mlops",
    )


# ==========================================================
# Header
# ==========================================================

st.title("⚙️ MachineGuard AI")

st.caption(
    "Industrial Machine Failure Prediction Platform"
)


# ==========================================================
# API Status
# ==========================================================

readiness = get_api_readiness()

prediction_enabled = False

if readiness is None:

    st.error("🔴 Backend Offline")

elif readiness.get("status") == "ready":

    prediction_enabled = True

    st.success(
        f"""
🟢 API Online

Model:
{readiness.get('model_name')}

Version:
{readiness.get('model_version')}
"""
    )

else:

    st.warning("🟡 Backend Running")

st.divider()


# ==========================================================
# Example Data
# ==========================================================

if st.button("Load Example Data"):

    st.session_state["air"] = 298.1
    st.session_state["process"] = 308.6
    st.session_state["speed"] = 1551.0
    st.session_state["torque"] = 42.8
    st.session_state["wear"] = 0.0


# ==========================================================
# Prediction Form
# ==========================================================

with st.form(
    key="prediction_form",
    clear_on_submit=False,
):

    st.subheader("Machine Measurements")

    left, right = st.columns(2)

    with left:

        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"],
            index=1,
        )

        air_temperature = st.number_input(
            "Air Temperature (K)",
            min_value=250.0,
            max_value=400.0,
            value=st.session_state.get(
                "air",
                298.1,
            ),
            step=0.1,
        )

        process_temperature = st.number_input(
            "Process Temperature (K)",
            min_value=250.0,
            max_value=450.0,
            value=st.session_state.get(
                "process",
                308.6,
            ),
            step=0.1,
        )

    with right:

        rotational_speed = st.number_input(
            "Rotational Speed (RPM)",
            min_value=0.0,
            max_value=5000.0,
            value=st.session_state.get(
                "speed",
                1551.0,
            ),
            step=1.0,
        )

        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            max_value=200.0,
            value=st.session_state.get(
                "torque",
                42.8,
            ),
            step=0.1,
        )

        tool_wear = st.number_input(
            "Tool Wear (minutes)",
            min_value=0.0,
            max_value=500.0,
            value=st.session_state.get(
                "wear",
                0.0,
            ),
            step=1.0,
        )

    submitted = st.form_submit_button(
        "Predict Failure Risk",
        type="primary",
        use_container_width=True,
        disabled=not prediction_enabled,
    )


# ==========================================================
# Prediction
# ==========================================================

if submitted:

    payload = {
        "machine_type": machine_type,
        "air_temperature": float(air_temperature),
        "process_temperature": float(process_temperature),
        "rotational_speed": float(rotational_speed),
        "torque": float(torque),
        "tool_wear": float(tool_wear),
    }

    try:

        with st.spinner(
            "Calculating failure risk..."
        ):
            result = submit_prediction(payload)

        display_prediction(result)

    except requests.Timeout:

        st.error(
            "Prediction request timed out."
        )

    except requests.HTTPError as error:

        response_text = ""

        if error.response is not None:
            response_text = error.response.text

        st.error(
            "The API rejected the request."
        )

        if response_text:
            st.code(
                response_text,
                language="json",
            )

    except requests.RequestException as error:

        st.error(
            f"Could not connect to the API:\n{error}"
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:

        st.error(
            f"Invalid API response:\n{error}"
        )


# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "MachineGuard AI • End-to-End MLOps Project • Built with FastAPI, MLflow, Docker, Streamlit and Render"
)