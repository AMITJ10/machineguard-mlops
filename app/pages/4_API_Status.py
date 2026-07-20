"""MachineGuard API Status."""

from __future__ import annotations

import os
from datetime import datetime

import requests
import streamlit as st

API_URL = os.getenv(
    "API_URL",
    "https://machineguard-mlops.onrender.com",
).rstrip("/")

TIMEOUT = 10

st.set_page_config(
    page_title="API Status",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ MachineGuard API Status")

st.caption("Monitor the health of the deployed FastAPI backend.")

if st.button("🔄 Refresh Status", use_container_width=True):
    st.rerun()


def check_endpoint(endpoint: str):
    """Call an API endpoint safely."""
    try:
        response = requests.get(
            f"{API_URL}{endpoint}",
            timeout=TIMEOUT,
        )

        return {
            "success": True,
            "status_code": response.status_code,
            "data": response.json(),
            "response_time": response.elapsed.total_seconds(),
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error),
        }


health = check_endpoint("/health")
ready = check_endpoint("/ready")

left, right = st.columns(2)

with left:
    st.subheader("❤️ Health")

    if health["success"]:
        st.success("Healthy")

        st.metric(
            "HTTP Status",
            health["status_code"],
        )

        st.metric(
            "Response Time",
            f"{health['response_time']:.3f} sec",
        )

        st.json(health["data"])

    else:
        st.error("Unavailable")
        st.code(health["error"])


with right:
    st.subheader("🚀 Readiness")

    if ready["success"]:
        st.success("Ready")

        data = ready["data"]

        st.metric(
            "Model Version",
            data.get("model_version", "-"),
        )

        st.metric(
            "Model Alias",
            data.get("model_alias", "-"),
        )

        st.metric(
            "Model Name",
            data.get("model_name", "-"),
        )

        st.json(data)

    else:
        st.error("Model Not Ready")
        st.code(ready["error"])


st.divider()

st.subheader("📡 API Information")

st.code(API_URL)

st.markdown(
    f"""
**Swagger Docs**

{API_URL}/docs

**OpenAPI**

{API_URL}/openapi.json

**Metrics**

{API_URL}/metrics

**Health**

{API_URL}/health

**Ready**

{API_URL}/ready
"""
)

st.divider()

st.subheader("⏱ Last Checked")

st.info(datetime.now().strftime("%d %b %Y %H:%M:%S"))