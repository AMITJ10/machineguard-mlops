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

# ---------------------------------------------------------
# Realistic operating ranges (narrower than the API's hard validation
# bounds). Values outside these are physically valid but unreliable —
# the field is flagged and the Predict button stays disabled until the
# reading is corrected.
# ---------------------------------------------------------

_RANGES = {
    "air": (295.0, 305.0, "Air Temperature (K)"),
    "process": (305.0, 315.0, "Process Temperature (K)"),
    "speed": (1150.0, 2900.0, "Rotational Speed (RPM)"),
    "torque": (3.0, 77.0, "Torque (Nm)"),
    "wear": (0.0, 253.0, "Tool Wear (minutes)"),
}


def _range_check(value: float, key: str) -> bool:
    """Return True if value is within the realistic operating range."""
    lo, hi, _label = _RANGES[key]
    return lo <= value <= hi


left, right = st.columns(2)

machine_type_labels = {
    "L": "L — Low grade (economy build, lower stress tolerance)",
    "M": "M — Medium grade (standard build)",
    "H": "H — High grade (heavy-duty, higher stress tolerance)",
}
machine_type_options = list(machine_type_labels.keys())

with left:

    machine_type = st.selectbox(
        "Machine Type",
        machine_type_options,
        index=machine_type_options.index(
            st.session_state.get("machine_type", "M")
        ),
        format_func=lambda code: machine_type_labels[code],
        help=(
            "Manufacturing quality grade of the machine. Lower grades "
            "have lower tolerance to combined wear and torque stress."
        ),
    )

    lo, hi, _ = _RANGES["air"]
    air_temperature = st.number_input(
        f"Air Temperature (K) — Valid Range: {lo:.0f} to {hi:.0f}",
        min_value=250.0,
        max_value=400.0,
        value=st.session_state.get("air", 298.1),
        step=0.1,
        key="air",
    )
    if not _range_check(air_temperature, "air"):
        st.markdown(f":red[⚠ Must be between {lo:.0f} and {hi:.0f} K.]")

    lo, hi, _ = _RANGES["process"]
    process_temperature = st.number_input(
        f"Process Temperature (K) — Valid Range: {lo:.0f} to {hi:.0f}",
        min_value=250.0,
        max_value=450.0,
        value=st.session_state.get("process", 308.6),
        step=0.1,
        key="process",
    )
    if not _range_check(process_temperature, "process"):
        st.markdown(f":red[⚠ Must be between {lo:.0f} and {hi:.0f} K.]")

with right:

    lo, hi, _ = _RANGES["speed"]
    rotational_speed = st.number_input(
        f"Rotational Speed (RPM) — Valid Range: {lo:.0f} to {hi:.0f}",
        min_value=0.0,
        max_value=5000.0,
        value=st.session_state.get("speed", 1551.0),
        step=1.0,
        key="speed",
    )
    if not _range_check(rotational_speed, "speed"):
        st.markdown(f":red[⚠ Must be between {lo:.0f} and {hi:.0f} RPM.]")

    lo, hi, _ = _RANGES["torque"]
    torque = st.number_input(
        f"Torque (Nm) — Valid Range: {lo:.0f} to {hi:.0f}",
        min_value=0.0,
        max_value=200.0,
        value=st.session_state.get("torque", 42.8),
        step=0.1,
        key="torque",
    )
    if not _range_check(torque, "torque"):
        st.markdown(f":red[⚠ Must be between {lo:.0f} and {hi:.0f} Nm.]")

    lo, hi, _ = _RANGES["wear"]
    tool_wear = st.number_input(
        f"Tool Wear (minutes) — Valid Range: {lo:.0f} to {hi:.0f}",
        min_value=0.0,
        max_value=500.0,
        value=st.session_state.get("wear", 0.0),
        step=1.0,
        key="wear",
    )
    if not _range_check(tool_wear, "wear"):
        st.markdown(f":red[⚠ Must be between {lo:.0f} and {hi:.0f} minutes.]")

all_valid = all(
    _range_check(value, key)
    for key, value in {
        "air": air_temperature,
        "process": process_temperature,
        "speed": rotational_speed,
        "torque": torque,
        "wear": tool_wear,
    }.items()
)

if not all_valid:
    st.warning(
        "One or more readings are outside the realistic operating range. "
        "Correct the highlighted fields to enable prediction."
    )

submitted = st.button(
    "🚀 Predict Failure",
    use_container_width=True,
    disabled=not all_valid,
)

# ---------------------------------------------------------
# Failure-mechanism analysis
#
# Each rule below maps to a documented failure mode used in industrial
# predictive-maintenance datasets. Since inputs are already validated
# to a realistic operating range before this point, findings here are
# genuine mechanical failure conditions, not sensor/data-quality issues.
# ---------------------------------------------------------

_OVERSTRAIN_LIMITS = {"L": 11_000, "M": 12_000, "H": 13_000}


def analyze_risk_factors(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Identify which physical failure mechanism is implicated."""

    findings: list[dict[str, Any]] = []

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
                "factor": "Heat Dissipation Failure",
                "severity": "high",
                "cause": (
                    f"The gap between process and air temperature is only "
                    f"{temp_diff:.1f} K (below the 8.6 K margin needed for "
                    f"adequate cooling), and rotational speed is {rpm:.0f} "
                    "RPM — too low to generate enough airflow across the "
                    "motor and bearing housings to carry heat away. Heat is "
                    "building up faster than the machine can dissipate it."
                ),
                "checklist": [
                    "Check cooling fan operation and airflow path for obstructions or dust buildup.",
                    "Verify coolant/lubricant flow rate and temperature at the heat exchanger.",
                    "Inspect bearing housings and motor windings for excessive heat with a thermal camera or IR thermometer.",
                    "Confirm ambient enclosure temperature isn't elevated by nearby equipment.",
                ],
                "action": (
                    "If safe for the process, increase spindle/rotational speed to restore airflow. "
                    "Clean or replace cooling fan filters, and verify coolant pump output. "
                    "If temperatures don't recover within one operating cycle, shut down and inspect bearings for heat damage before restarting."
                ),
            }
        )

    # --- Power Delivery Failure ---
    angular_velocity = 2.0 * math.pi * rpm / 60.0
    mechanical_power = torque_value * angular_velocity
    if mechanical_power < 3500 or mechanical_power > 9000:
        if mechanical_power < 3500:
            cause = (
                f"Computed mechanical power is only {mechanical_power:,.0f} W "
                "for this torque/speed combination — below the 3,500 W "
                "threshold for stable operation. This typically means the "
                "machine is running under-loaded relative to its rated "
                "torque, or the speed setpoint is too low for the torque "
                "being applied, which can cause the drive to hunt for a "
                "stable operating point and stress the motor controller."
            )
            action = (
                "Verify the torque and speed setpoints match the actual job requirement — "
                "an under-loaded combination is often a programming/setpoint error rather than "
                "a mechanical fault. Check the VFD/motor controller for undervoltage or fault codes."
            )
        else:
            cause = (
                f"Computed mechanical power is {mechanical_power:,.0f} W — "
                "above the 9,000 W threshold. The motor and drive train are "
                "being asked to deliver more power than the rated operating "
                "envelope, which accelerates wear on bearings, couplings, "
                "and the motor windings, and increases the risk of an "
                "overcurrent trip or motor burnout."
            )
            action = (
                "Reduce torque or speed setpoint immediately to bring power draw back within the rated "
                "envelope. Check the motor's current draw against its nameplate rating, and inspect the "
                "coupling/gearbox for signs of overload stress (unusual noise, vibration, heat)."
            )

        findings.append(
            {
                "factor": "Power Delivery Failure",
                "severity": "high",
                "cause": cause,
                "checklist": [
                    "Check the motor drive/VFD for fault codes or current draw outside rated limits.",
                    "Inspect the coupling, belt, or gearbox for slippage, wear, or misalignment.",
                    "Confirm torque and speed setpoints match the intended process, not a leftover test configuration.",
                    "Listen/feel for abnormal vibration or noise at the motor and drive train under load.",
                ],
                "action": action,
            }
        )

    # --- Overstrain Failure (Tool Wear x Torque) ---
    overstrain_score = wear * torque_value
    limit = _OVERSTRAIN_LIMITS.get(machine_type, 12_000)
    if overstrain_score > limit:
        findings.append(
            {
                "factor": "Overstrain Failure",
                "severity": "high",
                "cause": (
                    f"Combined tool wear and torque load ({overstrain_score:,.0f}) "
                    f"exceeds the {limit:,} limit rated for a type-{machine_type} "
                    "machine. A worn cutting tool needs more torque to remove "
                    "the same amount of material, and that extra mechanical "
                    "stress is transmitted through the spindle, tool holder, "
                    "and workpiece — raising the risk of tool breakage or "
                    "spindle damage."
                ),
                "checklist": [
                    "Inspect the cutting tool or tool tip for visible wear, chipping, or edge rounding.",
                    "Check tool holder/collet for looseness or runout.",
                    "Review recent tool-change logs — is this tool overdue for replacement?",
                    "Inspect the spindle for unusual vibration or noise under load.",
                ],
                "action": (
                    "Replace or re-sharpen the tool before continuing production. "
                    "Reduce feed rate or torque temporarily if the job must continue before a scheduled changeover. "
                    "Log the wear reading against the tool's rated service life to catch this earlier next time."
                ),
            }
        )

    # --- Tool Wear Failure ---
    if 200 <= wear <= 240:
        findings.append(
            {
                "factor": "Tool Wear Failure",
                "severity": "medium",
                "cause": (
                    f"Accumulated tool wear is {wear:.0f} minutes, inside the "
                    "200-240 minute band where random tool-failure incidents "
                    "cluster in this equipment class — this is close to the "
                    "statistical end of a typical tool's usable service life."
                ),
                "checklist": [
                    "Check tool wear against the manufacturer's rated tool life for this material/operation.",
                    "Visually inspect the cutting edge for micro-chipping or built-up edge.",
                    "Review the last few parts produced for dimensional drift or surface-finish degradation.",
                ],
                "action": (
                    "Schedule a tool change at the next planned stop rather than waiting for a failure. "
                    "If this tool is being pushed past its rated life regularly, consider adjusting the "
                    "preventive-maintenance interval for this tool type."
                ),
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

        if findings:
            severity_icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}

            for finding in findings:
                icon = severity_icon.get(finding["severity"], "🟡")

                with st.expander(f"{icon} {finding['factor']}", expanded=True):
                    st.markdown(f"**Likely cause:** {finding['cause']}")

                    st.markdown("**Inspection checklist:**")
                    for item in finding["checklist"]:
                        st.markdown(f"- {item}")

                    st.markdown(f"**Recommended action:** {finding['action']}")

        elif result["risk_level"] in ("high", "critical"):
            st.warning(
                "The model flagged elevated risk, but no single known "
                "failure mechanism explains it on its own — this can happen "
                "with an unusual combination of readings. Schedule a general "
                "inspection covering cooling, drive load, and tool condition "
                "rather than relying on one specific fix."
            )
        else:
            st.info(
                "No failure mechanism is indicated — readings are within "
                "normal operating bounds. Continue routine monitoring and "
                "scheduled maintenance."
            )