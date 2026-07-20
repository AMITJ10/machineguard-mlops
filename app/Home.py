"""MachineGuard AI - Home."""

from __future__ import annotations

import streamlit as st

from services.api import get_ready
from utils.styles import load_css

st.set_page_config(
    page_title="MachineGuard AI",
    page_icon="⚙️",
    layout="wide",
)

load_css()

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:

    st.image(
        "https://img.icons8.com/fluency/96/settings.png",
        width=70,
    )

    st.title("MachineGuard AI")

    st.caption(
        "Industrial Predictive Maintenance"
    )

    st.divider()

    st.subheader("Navigate")

    st.write(
        """
Go to the pages on the left.

• Single Prediction

• Batch Prediction

• Model Monitoring

• API Status

• About
"""
    )

    st.divider()

    st.link_button(
        "📘 API Docs",
        "https://machineguard-mlops.onrender.com/docs",
        use_container_width=True,
    )

    st.link_button(
        "💻 GitHub",
        "https://github.com/AMITJ10/machineguard-mlops",
        use_container_width=True,
    )

# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.title("⚙️ MachineGuard AI")

st.caption(
    "Predict industrial machine failures before they happen."
)

# ---------------------------------------------------
# Hero
# ---------------------------------------------------

left, right = st.columns([2, 1])

with left:

    st.markdown(
        """
### AI-powered Predictive Maintenance

MachineGuard AI analyzes machine sensor data and predicts
equipment failures before they occur.

Reduce downtime.

Reduce maintenance cost.

Increase equipment reliability.

Built using modern MLOps practices and deployed on the cloud.
"""
    )

    col1, col2 = st.columns(2)

    with col1:

        st.page_link(
            "pages/1_Single_Prediction.py",
            label="🚀 Start Prediction",
        )

    with col2:

        st.page_link(
            "pages/2_Batch_Prediction.py",
            label="📂 Batch Prediction",
        )

with right:

    try:

        ready = get_ready()

        if ready.get("status") == "ready":

            st.success("🟢 API Online")

        else:

            st.warning("🟡 API Starting")

    except Exception:

        st.error("🔴 API Offline")

# ---------------------------------------------------
# Features
# ---------------------------------------------------

st.divider()

st.subheader("Platform Features")

c1, c2, c3 = st.columns(3)

with c1:

    st.info(
        """
### 🤖 Machine Learning

Random Forest model

Probability prediction

Risk classification
"""
    )

with c2:

    st.info(
        """
### ☁️ Cloud Deployment

FastAPI

Docker

Render

Streamlit Cloud
"""
    )

with c3:

    st.info(
        """
### 📈 MLOps

MLflow

Monitoring

CI/CD

Model Registry
"""
    )

# ---------------------------------------------------
# How it Works
# ---------------------------------------------------

st.divider()

st.subheader("How It Works")

step1, step2, step3 = st.columns(3)

with step1:

    st.markdown(
        """
### 1️⃣ Enter Data

Provide machine sensor values
or upload a CSV.
"""
    )

with step2:

    st.markdown(
        """
### 2️⃣ AI Prediction

MachineGuard sends the
request to the prediction API.
"""
    )

with step3:

    st.markdown(
        """
### 3️⃣ Results

Receive:

• Failure probability

• Risk level

• Prediction
"""
    )

# ---------------------------------------------------
# Quick Navigation
# ---------------------------------------------------

st.divider()

st.subheader("Quick Navigation")

nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:

    st.page_link(
        "pages/1_Single_Prediction.py",
        label="Single Prediction",
    )

with nav2:

    st.page_link(
        "pages/2_Batch_Prediction.py",
        label="Batch Prediction",
    )

with nav3:

    st.page_link(
        "pages/4_API_Status.py",
        label="API Status",
    )

with nav4:

    st.page_link(
        "pages/5_About.py",
        label="About",
    )

# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.divider()

st.caption(
    "MachineGuard AI • Production MLOps Demo • FastAPI • MLflow • Docker • Render • Streamlit"
)