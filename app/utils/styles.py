"""Reusable Streamlit UI components."""

from __future__ import annotations

import streamlit as st


def load_css() -> None:
    """Load custom styles."""

    st.markdown(
        """
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.metric-card{

    border-radius:12px;

    padding:15px;

    background:#fafafa;

    border:1px solid #eeeeee;

}

</style>
""",
        unsafe_allow_html=True,
    )


def probability_bar(
    probability: float,
) -> None:
    """Display probability gauge."""

    st.progress(probability)

    st.caption(
        f"Failure Probability : {probability:.2%}"
    )


def risk_color(
    risk: str,
) -> None:
    """Risk message."""

    risk = risk.lower()

    if risk == "low":

        st.success(
            "✅ Low Risk"
        )

    elif risk == "medium":

        st.warning(
            "🟡 Medium Risk"
        )

    elif risk == "high":

        st.error(
            "🟠 High Risk"
        )

    else:

        st.error(
            "🔴 Critical Risk"
        )


def render_prediction_card(
    prediction: int,
    probability: float,
    risk: str,
) -> None:
    """Prediction dashboard."""

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Prediction",
            "Failure"
            if prediction == 1
            else "Healthy",
        )

    with c2:

        st.metric(
            "Probability",
            f"{probability:.2%}",
        )

    with c3:

        st.metric(
            "Risk",
            risk.upper(),
        )

    probability_bar(probability)

    risk_color(risk)