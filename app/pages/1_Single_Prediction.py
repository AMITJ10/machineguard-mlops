"""
MachineGuard - Single Prediction
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services.api import APIError, predict
from app.utils.styles import (
    load_css,
    probability_bar,
    risk_badge,
)

st.set_page_config(
    page_title="Single Prediction",
    page_icon="⚙️",
    layout="wide",
)

load_css()

# -----------------------------------------------------
# Header
# -----------------------------------------------------

st.title("⚙️ Single Machine Prediction")

st.caption(
    "Predict industrial machine failure using the deployed MachineGuard model."
)

st.divider()

# -----------------------------------------------------
# Example Data
# -----------------------------------------------------

DEFAULT_VALUES = {
    "machine_type": "M",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551.0,
    "torque": 42.8,
    "tool_wear": 0.0,
}

if "prediction_values" not in st.session_state:
    st.session_state.prediction_values = DEFAULT_VALUES.copy()

left, right = st.columns([1, 4])

with left:

    if st.button(
        "📋 Load Example Data",
        use_container_width=True,
    ):
        st.session_state.prediction_values = DEFAULT_VALUES.copy()
        st.rerun()

# -----------------------------------------------------
# Input Form
# -----------------------------------------------------

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"],
            index=["L", "M", "H"].index(
                st.session_state.prediction_values["machine_type"]
            ),
        )

        air_temperature = st.number_input(
            "Air Temperature (K)",
            value=float(
                st.session_state.prediction_values[
                    "air_temperature"
                ]
            ),
            step=0.1,
        )

        process_temperature = st.number_input(
            "Process Temperature (K)",
            value=float(
                st.session_state.prediction_values[
                    "process_temperature"
                ]
            ),
            step=0.1,
        )

    with col2:

        rotational_speed = st.number_input(
            "Rotational Speed (RPM)",
            value=float(
                st.session_state.prediction_values[
                    "rotational_speed"
                ]
            ),
            step=1.0,
        )

        torque = st.number_input(
            "Torque (Nm)",
            value=float(
                st.session_state.prediction_values["torque"]
            ),
            step=0.1,
        )

        tool_wear = st.number_input(
            "Tool Wear (minutes)",
            value=float(
                st.session_state.prediction_values[
                    "tool_wear"
                ]
            ),
            step=1.0,
        )

    submitted = st.form_submit_button(
        "🚀 Predict Failure",
        use_container_width=True,
    )

# -----------------------------------------------------
# Prediction
# -----------------------------------------------------

if submitted:

    payload = {
        "machine_type": machine_type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
    }

    try:

        with st.spinner("Running prediction..."):

            result = predict(payload)

        st.success("Prediction completed successfully.")

        probability = float(
            result["failure_probability"]
        )

        prediction = int(
            result["prediction"]
        )

        risk = result["risk_level"]

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Failure Probability",
                f"{probability:.2%}",
            )

        with col2:

            st.metric(
                "Prediction",
                "Failure"
                if prediction == 1
                else "No Failure",
            )

        with col3:

            st.metric(
                "Risk Level",
                risk.upper(),
            )

        probability_bar(probability)

        risk_badge(risk)

        st.divider()

        st.subheader("Prediction Response")

        st.json(result)

        st.divider()

        st.subheader("Download Result")

        dataframe = pd.DataFrame([result])

        csv = dataframe.to_csv(
            index=False,
        )

        st.download_button(
            "⬇ Download Prediction",
            csv,
            file_name="prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except APIError as error:

        st.error(str(error))

    except Exception as error:

        st.exception(error)