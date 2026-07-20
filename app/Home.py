"""MachineGuard AI Dashboard."""

from __future__ import annotations

import os

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


def api_status() -> bool:
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


# ======================================================
# Sidebar
# ======================================================

with st.sidebar:

    st.title("⚙️ MachineGuard AI")

    st.write(
        """
Industrial Predictive Maintenance Platform
built with modern MLOps.
"""
    )

    st.divider()

    st.subheader("Project Stack")

    st.markdown(
        """
- FastAPI
- Scikit-Learn
- MLflow
- Docker
- Render
- Streamlit
- AWS (Next Phase)
- Airflow (Next Phase)
"""
    )

    st.divider()

    st.subheader("Links")

    st.link_button(
        "📄 API Docs",
        "https://machineguard-mlops.onrender.com/docs",
        use_container_width=True,
    )

    st.link_button(
        "🌐 Backend API",
        "https://machineguard-mlops.onrender.com",
        use_container_width=True,
    )

    st.link_button(
        "💻 GitHub Repository",
        "https://github.com/AMITJ10/machineguard-mlops",
        use_container_width=True,
    )

# ======================================================
# Header
# ======================================================

st.title("⚙️ MachineGuard AI")

st.caption(
    "Industrial Machine Failure Prediction Platform"
)

st.divider()

# ======================================================
# API STATUS
# ======================================================

if api_status():

    st.success("🟢 Backend API Online")

else:

    st.error("🔴 Backend API Offline")

# ======================================================
# Overview
# ======================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "Random Forest",
    )

with col2:
    st.metric(
        "Deployment",
        "Render",
    )

with col3:
    st.metric(
        "Frontend",
        "Streamlit",
    )

st.divider()

st.subheader("Project Overview")

st.write(
    """
MachineGuard AI predicts industrial machine failures
using machine sensor measurements.

The project demonstrates a complete production-ready
Machine Learning Operations (MLOps) workflow including:

- Data Validation
- Feature Engineering
- Scikit-Learn Pipeline
- MLflow Experiment Tracking
- Model Registry
- FastAPI Prediction API
- Docker Deployment
- CI/CD with GitHub Actions
- Render Cloud Deployment
- Streamlit Dashboard
- Monitoring & Drift Detection
- AWS Cloud Storage (coming next)
- Airflow Retraining Pipeline (coming next)
"""
)

st.divider()

st.subheader("Navigation")

st.info(
    """
Use the left sidebar to access:

• Single Prediction

• Batch Prediction

• API Status
"""
)

st.divider()

st.caption(
    "MachineGuard AI • End-to-End MLOps Project • 2026"
)