"""MachineGuard API Status."""

from __future__ import annotations

import os
from datetime import datetime

import requests
import streamlit as st

st.set_page_config(
    page_title="API Status",
    page_icon="🛰️",
    layout="wide",
)

API_URL = os.getenv(
    "API_URL",
    "https://machineguard-mlops.onrender.com",
).rstrip("/")

TIMEOUT = 10


def check_endpoint(endpoint: str):
    """Safely call an API endpoint."""

    try:
        response = requests.get(
            f"{API_URL}{endpoint}",
            timeout=TIMEOUT,
        )

        response.raise_for_status()

        return {
            "success": True,
            "status_code": response.status_code,
            "response_time": response.elapsed.total_seconds(),
            "data": response.json(),
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error),
        }


st.title("🛰️ API Status")

st.caption(
    "Real-time health monitoring for the deployed MachineGuard API."
)

if st.button(
    "🔄 Refresh",
    use_container_width=True,
):
    st.rerun()

health = check_endpoint("/health")
ready = check_endpoint("/ready")

st.divider()

# ---------------------------------------------------------
# Overall Status
# ---------------------------------------------------------

status1, status2, status3 = st.columns(3)

with status1:

    if health["success"]:
        st.success("Backend Online")
    else:
        st.error("Backend Offline")

with status2:

    if ready["success"]:
        st.success("Prediction Service Ready")
    else:
        st.warning("Prediction Service Unavailable")

with status3:

    if health["success"]:
        st.metric(
            "Response Time",
            f"{health['response_time']:.3f}s",
        )
    else:
        st.metric(
            "Response Time",
            "-",
        )

st.divider()

# ---------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------

st.subheader("❤️ Health Endpoint")

if health["success"]:

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "HTTP Status",
            health["status_code"],
        )

    with c2:

        st.metric(
            "Latency",
            f"{health['response_time']:.3f} sec",
        )

    st.success("Backend is healthy.")

    st.json(health["data"])

else:

    st.error("Unable to connect.")

    st.code(health["error"])

st.divider()

# ---------------------------------------------------------
# Ready Endpoint
# ---------------------------------------------------------

st.subheader("🚀 Model Readiness")

if ready["success"]:

    data = ready["data"]

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Model Name",
            data.get(
                "model_name",
                "-",
            ),
        )

    with col2:

        st.metric(
            "Version",
            data.get(
                "model_version",
                "-",
            ),
        )

    with col3:

        st.metric(
            "Alias",
            data.get(
                "model_alias",
                "-",
            ),
        )

    st.success("Champion model loaded successfully.")

    st.json(data)

else:

    st.warning("Model not ready.")

    st.code(ready["error"])

st.divider()

# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

st.subheader("Available Endpoints")

endpoint_data = {
    "Endpoint": [
        "/health",
        "/ready",
        "/predict",
        "/docs",
        "/openapi.json",
        "/metrics",
    ],
    "Purpose": [
        "Health Check",
        "Model Readiness",
        "Single Prediction",
        "Swagger UI",
        "OpenAPI Schema",
        "Prometheus Metrics",
    ],
}

st.table(endpoint_data)

st.divider()

# ---------------------------------------------------------
# Deployment Information
# ---------------------------------------------------------

st.subheader("Deployment")

left, right = st.columns(2)

with left:

    st.info(
        f"""
**Backend URL**

{API_URL}

**Environment**

Production

**Hosting**

Render
"""
    )

with right:

    st.info(
        """
**Frontend**

Streamlit Cloud

**Container**

Docker

**Framework**

FastAPI
"""
    )

st.divider()

# ---------------------------------------------------------
# Quick Links
# ---------------------------------------------------------

st.subheader("Quick Links")

c1, c2, c3 = st.columns(3)

with c1:

    st.link_button(
        "📄 Swagger Docs",
        f"{API_URL}/docs",
        use_container_width=True,
    )

with c2:

    st.link_button(
        "🌐 Backend API",
        API_URL,
        use_container_width=True,
    )

with c3:

    st.link_button(
        "💻 GitHub",
        "https://github.com/AMITJ10/machineguard-mlops",
        use_container_width=True,
    )

st.divider()

# ---------------------------------------------------------
# Last Checked
# ---------------------------------------------------------

st.caption(
    f"Last checked: {datetime.now().strftime('%d %b %Y %H:%M:%S')}"
)

st.caption(
    "MachineGuard AI • Production API Monitoring"
)