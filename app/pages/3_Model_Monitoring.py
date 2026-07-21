"""MachineGuard Model Monitoring Dashboard."""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Model Monitoring",
    page_icon="📊",
    layout="wide",
)

API_URL = os.getenv(
    "API_URL",
    "https://machineguard-mlops.onrender.com",
).rstrip("/")

TIMEOUT = 10


# ----------------------------------------------------
# API HELPERS
# ----------------------------------------------------


def api_get(endpoint: str):
    """Safely call API endpoint."""

    try:
        response = requests.get(
            f"{API_URL}{endpoint}",
            timeout=TIMEOUT,
        )

        response.raise_for_status()

        return response.json()

    except Exception:
        return None


def load_health():
    return api_get("/health")


def load_ready():
    return api_get("/ready")


def load_metrics():
    return api_get("/metrics")


# ----------------------------------------------------
# HEADER
# ----------------------------------------------------

st.title("📊 Model Monitoring")

st.caption(
    "Production monitoring dashboard for MachineGuard AI."
)

if st.button(
    "🔄 Refresh Dashboard",
    use_container_width=True,
):
    st.rerun()

health = load_health()
ready = load_ready()

st.divider()

# ----------------------------------------------------
# SERVICE STATUS
# ----------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    if health:
        st.success("Backend Online")
    else:
        st.error("Backend Offline")

with col2:

    if ready:
        st.success("Model Loaded")
    else:
        st.warning("Model Not Ready")

with col3:

    if ready:
        st.info(
            ready.get(
                "model_version",
                "Unknown",
            )
        )
    else:
        st.info("-")

st.divider()

# ----------------------------------------------------
# MODEL INFORMATION
# ----------------------------------------------------

st.subheader("Current Production Model")

if ready:

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Model Name",
            ready.get(
                "model_name",
                "-",
            ),
        )

    with c2:
        st.metric(
            "Version",
            ready.get(
                "model_version",
                "-",
            ),
        )

    with c3:
        st.metric(
            "Alias",
            ready.get(
                "model_alias",
                "-",
            ),
        )

else:

    st.warning(
        "Production model information unavailable."
    )

st.divider()

# ----------------------------------------------------
# API HEALTH
# ----------------------------------------------------

st.subheader("API Health")

if health:

    st.success("API responding successfully.")

    st.json(health)

else:

    st.error(
        "Could not reach the deployed API."
    )

st.divider()

# ----------------------------------------------------
# PROMETHEUS METRICS
# ----------------------------------------------------

st.subheader("Prometheus Metrics")

metrics = load_metrics()

if metrics:

    st.success("Metrics endpoint available.")

    if isinstance(metrics, dict):

        st.json(metrics)

    else:

        st.code(
            str(metrics)[:3000]
        )

else:

    st.info(
        "Metrics endpoint is not enabled yet."
    )

st.divider()

# ----------------------------------------------------
# SAMPLE MONITORING METRICS
# ----------------------------------------------------

st.subheader("Runtime Statistics")

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Backend",
    "Online" if health else "Offline",
)

metric2.metric(
    "Model",
    "Loaded" if ready else "Unavailable",
)

metric3.metric(
    "Environment",
    "Production",
)

metric4.metric(
    "Platform",
    "Render",
)

st.divider()

# ----------------------------------------------------
# SAMPLE PREDICTION ANALYTICS
# ----------------------------------------------------

st.subheader("Prediction Analytics")

history = pd.DataFrame(
    {
        "Prediction": [
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
        ],
        "Failure Probability": [
            0.08,
            0.91,
            0.12,
            0.27,
            0.83,
            0.21,
            0.97,
            0.15,
            0.31,
            0.74,
        ],
    }
)

left, right = st.columns(2)

with left:

    st.markdown("##### Failure Probability")

    st.line_chart(
        history["Failure Probability"]
    )

with right:

    risk = history["Prediction"].value_counts()

    risk.index = [
        "Healthy",
        "Failure",
    ]

    st.markdown("##### Prediction Distribution")

    st.bar_chart(risk)

st.divider()

# ----------------------------------------------------
# DEPLOYMENT INFORMATION
# ----------------------------------------------------

# st.subheader("Deployment")

# col1, col2 = st.columns(2)

# with col1:

#     st.info(
#         f"""
# **Backend API**

# {API_URL}

# **Frontend**

# Streamlit Cloud

# **Cloud**

# Render
# """
#     )

# with col2:

#     st.info(
#         """
# **ML Framework**

# Scikit-Learn

# **Experiment Tracking**

# MLflow

# **Containerization**

# Docker
# """
#     )

# st.divider()

# ----------------------------------------------------
# ROADMAP
# ----------------------------------------------------

# st.subheader("Upcoming Production Features")

# roadmap = pd.DataFrame(
#     {
#         "Feature": [
#             "AWS S3 Model Registry",
#             "Grafana Dashboard",
#             "Prometheus Monitoring",
#             "Airflow Retraining",
#             "Drift Detection",
#             "Alerting",
#         ],
#         "Status": [
#             "Next Phase",
#             "Planned",
#             "Planned",
#             "Planned",
#             "Completed",
#             "Planned",
#         ],
#     }
# )

# st.dataframe(
#     roadmap,
#     use_container_width=True,
#     hide_index=True,
# )

# st.divider()

# ----------------------------------------------------
# SYSTEM INFORMATION
# ----------------------------------------------------

st.subheader("System Information")

info = pd.DataFrame(
    {
        "Component": [
            "Frontend",
            "Backend",
            "Model",
            "Deployment",
            "Environment",
        ],
        "Value": [
            "Streamlit",
            "FastAPI",
            (
                ready.get(
                    "model_name",
                    "-"
                )
                if ready
                else "-"
            ),
            "Render",
            "Production",
        ],
    }
)

st.table(info)

st.divider()

# ----------------------------------------------------
# QUICK LINKS
# ----------------------------------------------------

st.subheader("Resources")

c1, c2, c3 = st.columns(3)

with c1:

    st.link_button(
        "API Documentation",
        f"{API_URL}/docs",
        use_container_width=True,
    )

with c2:

    st.link_button(
        "Backend API",
        API_URL,
        use_container_width=True,
    )

with c3:

    st.link_button(
        "GitHub Repository",
        "https://github.com/AMITJ10/machineguard-mlops",
        use_container_width=True,
    )

st.divider()

# ----------------------------------------------------
# LAST UPDATED
# ----------------------------------------------------

st.caption(
    f"Dashboard refreshed on {datetime.now().strftime('%d %b %Y %H:%M:%S')}"
)

st.caption(
    "MachineGuard AI • Production Monitoring Dashboard"
)
