import streamlit as st


def risk_color(risk: str):
    risk = risk.lower()

    if risk == "low":
        st.success("🟢 Low Risk")

    elif risk == "medium":
        st.warning("🟡 Medium Risk")

    elif risk == "high":
        st.error("🟠 High Risk")

    else:
        st.error("🔴 Critical Risk")


def probability_bar(probability: float):
    st.progress(probability)
    st.caption(f"{probability:.2%} probability")