"""
MachineGuard Streamlit styling.
"""

from __future__ import annotations

import streamlit as st


def load_css() -> None:
    """
    Inject custom CSS.
    """

    st.markdown(
        """
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

/* Sidebar */

section[data-testid="stSidebar"]{
    background:#fafafa;
}

/* Metric Cards */

div[data-testid="metric-container"]{

    background:white;

    border-radius:14px;

    border:1px solid #e9ecef;

    padding:18px;

    box-shadow:0 2px 8px rgba(0,0,0,.08);
}

/* Buttons */

.stButton>button{

    width:100%;

    border-radius:10px;

    height:48px;

    font-weight:600;

}

/* Success */

.success-card{

    background:#ecfdf5;

    border-left:6px solid #16a34a;

    padding:18px;

    border-radius:10px;

}

/* Error */

.error-card{

    background:#fff5f5;

    border-left:6px solid #dc2626;

    padding:18px;

    border-radius:10px;

}

/* Warning */

.warning-card{

    background:#fff8e6;

    border-left:6px solid orange;

    padding:18px;

    border-radius:10px;

}

/* Info */

.info-card{

    background:#eef6ff;

    border-left:6px solid #2563eb;

    padding:18px;

    border-radius:10px;

}

.big-title{

    font-size:42px;

    font-weight:700;

}

.subtitle{

    color:#6c757d;

    margin-bottom:25px;

}

</style>
""",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------
# Cards
# ----------------------------------------------------------


def success_card(message: str):

    st.markdown(
        f"""
<div class="success-card">

{message}

</div>
""",
        unsafe_allow_html=True,
    )


def error_card(message: str):

    st.markdown(
        f"""
<div class="error-card">

{message}

</div>
""",
        unsafe_allow_html=True,
    )


def warning_card(message: str):

    st.markdown(
        f"""
<div class="warning-card">

{message}

</div>
""",
        unsafe_allow_html=True,
    )


def info_card(message: str):

    st.markdown(
        f"""
<div class="info-card">

{message}

</div>
""",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------
# Risk Colors
# ----------------------------------------------------------


def risk_badge(
    risk: str,
):
    """
    Colored badge for prediction risk.
    """

    risk = risk.lower()

    if risk == "low":
        st.success("🟢 LOW RISK")

    elif risk == "medium":
        st.warning("🟡 MEDIUM RISK")

    elif risk == "high":
        st.error("🟠 HIGH RISK")

    else:
        st.error("🔴 CRITICAL RISK")


def probability_bar(
    probability: float,
):
    """
    Prediction probability bar.
    """

    st.progress(float(probability))

    st.caption(
        f"Failure Probability: {probability:.2%}"
    )