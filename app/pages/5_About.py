"""About MachineGuard AI."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide",
)

st.title("ℹ️ About MachineGuard AI")

st.caption(
    "Industrial Predictive Maintenance Platform powered by Machine Learning and MLOps."
)

st.divider()

# ============================================================
# What is MachineGuard
# ============================================================

st.header("What is MachineGuard AI?")

st.write(
    """
MachineGuard AI is an end-to-end Machine Learning Operations (MLOps)
application that predicts industrial machine failures using sensor
measurements collected from manufacturing equipment.

The objective is to identify machines at risk of failure before
unexpected downtime occurs, allowing maintenance teams to take
preventive action.
"""
)

st.divider()

# ============================================================
# Key Features
# ============================================================

st.header("Key Features")

col1, col2 = st.columns(2)

with col1:

    st.success("✅ Real-time Failure Prediction")

    st.success("✅ Batch CSV Predictions")

    st.success("✅ REST API")

    st.success("✅ Interactive Dashboard")

    st.success("✅ Production Deployment")

with col2:

    st.success("✅ MLflow Model Registry")

    st.success("✅ Docker Container")

    st.success("✅ GitHub Actions CI")

    st.success("✅ Monitoring Dashboard")

    st.success("✅ Cloud Deployment")

st.divider()

# ============================================================
# Architecture
# ============================================================

st.header("Solution Architecture")

st.markdown(
    """
"""
)

st.divider()

# ============================================================
# Technology Stack
# ============================================================

st.header("Technology Stack")

stack1, stack2, stack3 = st.columns(3)

with stack1:

    st.subheader("Machine Learning")

    st.markdown(
        """
- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib
"""
    )

with stack2:

    st.subheader("MLOps")

    st.markdown(
        """
- MLflow
- Docker
- FastAPI
- GitHub Actions
- Render
"""
    )

with stack3:

    st.subheader("Frontend")

    st.markdown(
        """
- Streamlit
- Plotly
- Requests
"""
    )

st.divider()

# ============================================================
# Current Deployment
# ============================================================

st.header("Deployment")

left, right = st.columns(2)

with left:

    st.info(
        """
**Backend**

• FastAPI

• Docker

• Render Cloud
"""
    )

with right:

    st.info(
        """
**Frontend**

• Streamlit

• Connected to Production API
"""
    )

st.divider()

# ============================================================
# Roadmap
# ============================================================

st.header("Project Roadmap")

roadmap = [
    ("✅", "Data Validation"),
    ("✅", "Training Pipeline"),
    ("✅", "MLflow Experiment Tracking"),
    ("✅", "Model Registry"),
    ("✅", "FastAPI REST API"),
    ("✅", "Docker"),
    ("✅", "GitHub Actions CI/CD"),
    ("✅", "Render Deployment"),
    ("✅", "Streamlit Dashboard"),
    ("🟡", "AWS S3 Model Storage"),
    ("🟡", "Prometheus Monitoring"),
    ("🟡", "Grafana Dashboards"),
    ("🟡", "Airflow Retraining Pipeline"),
    ("🟡", "Vercel Landing Page"),
]

for status, item in roadmap:
    st.write(f"{status} {item}")

st.divider()

# ============================================================
# Open Source
# ============================================================

st.header("Source Code")

st.write(
    "MachineGuard AI is developed as a complete end-to-end MLOps portfolio project."
)

col1, col2 = st.columns(2)

with col1:

    st.link_button(
        "💻 GitHub Repository",
        "https://github.com/AMITJ10/machineguard-mlops",
        use_container_width=True,
    )

with col2:

    st.link_button(
        "📄 API Documentation",
        "https://machineguard-mlops.onrender.com/docs",
        use_container_width=True,
    )

st.divider()

# ============================================================
# Footer
# ============================================================

st.caption(
    "MachineGuard AI • End-to-End MLOps Portfolio Project • © 2026"
)