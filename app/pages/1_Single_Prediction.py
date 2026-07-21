"""Single Machine Prediction."""

from __future__ import annotations

import math
from typing import Any

import streamlit as st

from services.api import predict
from utils.styles import probability_bar, risk_color

st.set_page_config(
    page_title="Single Prediction",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Single Machine Prediction")

st.caption("Predict failure risk for one industrial machine.")

if st.button("📂 Load Example Data", use_container_width=True):
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
                st.session_state.get("machine_type", "M")
            ),
            help="L = low quality, M = medium quality, H = high quality.",
        )

        air_temperature = st.number_input(
            "Air Temperature (K)",
            value=st.session_state.get("air", 298.1),
            step=0.1,
            help="Typical training range is roughly 295-305 K.",
        )

        process_temperature = st.number_input(
            "Process Temperature (K)",
            value=st.session_state.get("process", 308.6),
            step=0.1,
            help="Typical training range is roughly 305-314 K.",
        )

    with right:

        rotational_speed = st.number_input(
            "Rotational Speed (RPM)",
            value=st.session_state.get("speed", 1551.0),
            step=1.0,
            help="Typical training range is roughly 1170-2900 RPM.",
        )

        torque = st.number_input(
            "Torque (Nm)",
            value=st.session_state.get("torque", 42.8),
            step=0.1,
            help="Typical training range is roughly 3-77 Nm.",
        )

        tool_wear = st.number_input(
            "Tool Wear (minutes)",
            value=st.session_state.get("wear", 0.0),
            step=1.0,
            help="Typical training range is roughly 0-250 minutes.",
        )

    submitted = st.form_submit_button(
        "🚀 Predict Failure",
        use_container_width=True,
    )

# ---------------------------------------------------------
# Rule-based explanation of *why* a prediction is high/low risk.
#
# These thresholds mirror the known failure mechanisms used to label
# this style of predictive-maintenance dataset (heat dissipation,
# power, overstrain, and tool-wear failure modes). The trained model
# has learned these patterns from data; this panel makes the same
# logic visible in plain language so the "critical risk" verdict is
# explainable and actionable, not just a probability.
# ---------------------------------------------------------

# Overstrain failure threshold (tool_wear * torque) depends on machine type.
_OVERSTRAIN_LIMITS = {"L": 11_000, "M": 12_000, "H": 13_000}


def analyze_risk_factors(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Identify which physical failure mechanisms this input is close to."""

    findings: list[dict[str, str]] = []

    air_temp = payload["air_temperature"]
    process_temp = payload["process_temperature"]
    rpm = payload["rotational_speed"]
    torque_value = payload["torque"]
    wear = payload["tool_wear"]
    machine_type = payload["machine_type"]

    # --- Heat Dissipation Failure ---
    temp_diff = process_temp - air_temp
    if temp_diff < 8.6 and rpm < 1380:
        findings.append(
            {
                "factor": "Heat Dissipation",
                "severity": "high",
                "detail": (
                    f"Process/air temperature gap is only {temp_diff:.1f} K "
                    f"(below 8.6 K) while rotational speed is {rpm:.0f} RPM "
                    "(below 1380). Low airflow at low speed is limiting heat "
                    "dissipation."
                ),
                "action": (
                    "Increase spindle speed if the process allows it, check "
                    "for blocked cooling airflow, and verify coolant or fan "
                    "operation."
                ),
            }
        )

    # --- Power Failure ---
    angular_velocity = 2.0 * math.pi * rpm / 60.0
    mechanical_power = torque_value * angular_velocity
    if mechanical_power < 3500 or mechanical_power > 9000:
        direction = "below the 3,500 W minimum" if mechanical_power < 3500 else "above the 9,000 W maximum"
        findings.append(
            {
                "factor": "Power Delivery",
                "severity": "high",
                "detail": (
                    f"Computed mechanical power is {mechanical_power:,.0f} W, "
                    f"{direction} for stable operation given this torque/RPM "
                    "combination."
                ),
                "action": (
                    "Re-check torque and speed setpoints together — they may "
                    "be an unrealistic combination for this machine — or "
                    "inspect the drive/motor for under- or over-powering."
                ),
            }
        )

    # --- Overstrain Failure ---
    overstrain_score = wear * torque_value
    limit = _OVERSTRAIN_LIMITS.get(machine_type, 12_000)
    if overstrain_score > limit:
        findings.append(
            {
                "factor": "Overstrain (Tool Wear × Torque)",
                "severity": "high",
                "detail": (
                    f"Tool wear × torque = {overstrain_score:,.0f}, above the "
                    f"{limit:,} limit for a type-{machine_type} machine."
                ),
                "action": (
                    "Schedule tool replacement soon and reduce torque load "
                    "if possible until the tool is swapped."
                ),
            }
        )

    # --- Tool Wear Failure ---
    if 200 <= wear <= 240:
        findings.append(
            {
                "factor": "Tool Wear",
                "severity": "medium",
                "detail": (
                    f"Tool wear is {wear:.0f} minutes, inside the 200-240 "
                    "minute band where tool failures cluster."
                ),
                "action": "Plan a tool inspection/replacement in the next maintenance window.",
            }
        )

    # --- Out-of-range inputs (extrapolation warning) ---
    if not (250 <= air_temp <= 320) or not (250 <= process_temp <= 330):
        findings.append(
            {
                "factor": "Unusual Temperature Reading",
                "severity": "low",
                "detail": (
                    "Air or process temperature is well outside the range "
                    "the model was trained on (roughly 295-315 K). "
                    "Predictions on out-of-range sensor data extrapolate "
                    "and can be overconfident."
                ),
                "action": "Double-check the sensor reading and unit (Kelvin) before acting on this prediction.",
            }
        )

    if not (500 <= rpm <= 3000):
        findings.append(
            {
                "factor": "Unusual Rotational Speed",
                "severity": "low",
                "detail": (
                    f"{rpm:.0f} RPM is outside the typical operating range "
                    "the model was trained on (roughly 1170-2900 RPM)."
                ),
                "action": "Confirm this speed reading is correct for the machine before trusting the prediction.",
            }
        )

    return findings


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
        probability = float(result["failure_probability"])

        st.success("Prediction Completed")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Failure Probability", f"{probability:.2%}")

        with c2:
            st.metric(
                "Prediction",
                "Failure" if result["prediction"] == 1 else "Healthy",
            )

        with c3:
            st.metric("Risk", result["risk_level"])

        probability_bar(probability)
        risk_color(result["risk_level"])

        # -----------------------------------------------
        # Focus Areas / Preventive Actions
        # -----------------------------------------------

        st.divider()
        st.subheader("🔍 Focus Areas & Preventive Actions", anchor=False)

        findings = analyze_risk_factors(payload)

        if not findings:
            st.info(
                "No single failure mechanism stands out — readings look "
                "within normal operating bounds. Continue routine monitoring."
            )
        else:
            severity_icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}

            for finding in findings:
                icon = severity_icon.get(finding["severity"], "🟡")

                with st.expander(f"{icon} {finding['factor']}", expanded=True):
                    st.write(finding["detail"])
                    st.markdown(f"**Recommended action:** {finding['action']}")

        with st.expander("API Response"):
            st.json(result)