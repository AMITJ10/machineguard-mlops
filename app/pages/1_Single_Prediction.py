"""Single Machine Prediction."""

from __future__ import annotations

import streamlit as st

from services.api import predict
from utils.styles import probability_bar, risk_color

st.set_page_config(
    page_title="Single Prediction",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Single Machine Prediction")

st.caption(
    "Predict failure risk for one industrial machine."
)

if st.button(
    "📂 Load Example Data",
    use_container_width=True,
):
    st.session_state["machine_type"] = "M"
    st.session_state["air"] = 298.1
    st.session_state["process"] = 308.6
    st.session_state["speed"] = 1551.0
    st.session_state["torque"] = 42.8
    st.session_state["wear"] = 0.0

with st.form("prediction_form"):

    left, right = st.columns(2)

    with left:

        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"],
            index=["L", "M", "H"].index(
                st.session_state.get(
                    "machine_type",
                    "M",
                )
            ),
        )

        air_temperature = st.number_input(
            "Air Temperature (K)",
            value=st.session_state.get(
                "air",
                298.1,
            ),
            step=0.1,
        )

        process_temperature = st.number_input(
            "Process Temperature (K)",
            value=st.session_state.get(
                "process",
                308.6,
            ),
            step=0.1,
        )

    with right:

        rotational_speed = st.number_input(
            "Rotational Speed (RPM)",
            value=st.session_state.get(
                "speed",
                1551.0,
            ),
            step=1.0,
        )

        torque = st.number_input(
            "Torque (Nm)",
            value=st.session_state.get(
                "torque",
                42.8,
            ),
            step=0.1,
        )

        tool_wear = st.number_input(
            "Tool Wear (minutes)",
            value=st.session_state.get(
                "wear",
                0.0,
            ),
            step=1.0,
        )

    submitted = st.form_submit_button(
        "🚀 Predict Failure",
        use_container_width=True,
    )

if submitted:

    payload = {
        "machine_type": machine_type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
    }

    with st.spinner("Running prediction..."):

        result = predict(payload)

    if result is None:

        st.error("Prediction failed.")

    else:

        probability = float(
            result["failure_probability"]
        )

        st.success("Prediction Completed")

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
                if result["prediction"] == 1
                else "Healthy",
            )

        with c3:
            st.metric(
                "Risk",
                result["risk_level"],
            )

        probability_bar(probability)

        risk_color(
            result["risk_level"]
        )

        with st.expander(
            "API Response",
        ):
            st.json(result)