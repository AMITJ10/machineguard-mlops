"""MachineGuard AI - Home."""

from __future__ import annotations

from services.api import get_ready
from utils.styles import load_css

import streamlit as st

st.set_page_config(
    page_title="MachineGuard AI",
    page_icon="⚙️",
    layout="wide",
)

load_css()

# ---------------------------------------------------
# Extra dashboard CSS (on top of load_css())
# ---------------------------------------------------

st.markdown(
    """
    <style>
    .mg-hero {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border-radius: 18px;
        padding: 2.2rem 2.4rem;
        color: #f9fafb;
        margin-bottom: 1.4rem;
    }
    .mg-hero h1 {
        margin: 0 0 0.4rem 0;
        font-size: 2.1rem;
    }
    .mg-hero p {
        color: #d1d5db;
        font-size: 1.02rem;
        margin: 0;
    }
    .mg-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
    }
    .mg-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    }
    .mg-card h4 {
        margin: 0 0 0.5rem 0;
    }
    .mg-card p {
        color: #6b7280;
        font-size: 0.92rem;
        margin: 0;
    }
    .mg-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .mg-badge-green { background: #dcfce7; color: #166534; }
    .mg-badge-yellow { background: #fef9c3; color: #854d0e; }
    .mg-badge-red { background: #fee2e2; color: #991b1b; }
    .mg-activity-row {
        display: flex;
        justify-content: space-between;
        padding: 0.55rem 0.2rem;
        border-bottom: 1px solid #f0f0f0;
        font-size: 0.9rem;
    }
    .mg-section-title {
        margin-top: 0.4rem;
        margin-bottom: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/settings.png",
        width=70,
    )

    st.title("MachineGuard AI")
    st.caption("Industrial Predictive Maintenance")

    st.divider()
    st.subheader("Navigate")

    st.page_link("pages/1_Single_Prediction.py", label="🚀 Single Prediction")
    st.page_link("pages/2_Batch_Prediction.py", label="📂 Batch Prediction")
    st.page_link("pages/3_Model_Monitoring.py", label="📊 Model Monitoring")
    st.page_link("pages/4_API_Status.py", label="🔌 API Status")
    st.page_link("pages/5_About.py", label="ℹ️ About")

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
# Check API status once, reuse everywhere on the page
# ---------------------------------------------------

api_status = "offline"
api_detail = ""

try:
    ready = get_ready()
    api_status = "ready" if ready.get("status") == "ready" else "starting"
    api_detail = str(ready.get("status", ""))
except Exception as exc:  # noqa: BLE001
    api_status = "offline"
    api_detail = str(exc)

status_badge = {
    "ready": ("🟢 API Online", "mg-badge-green"),
    "starting": ("🟡 API Starting", "mg-badge-yellow"),
    "offline": ("🔴 API Offline", "mg-badge-red"),
}[api_status]

# ---------------------------------------------------
# Hero
# ---------------------------------------------------

st.markdown(
    f"""
    <div class="mg-hero">
        <h1>⚙️ MachineGuard AI</h1>
        <p>Predict industrial machine failures before they happen — powered by
        machine learning, deployed with production MLOps practices.</p>
        <br>
        <span class="mg-badge {status_badge[1]}">{status_badge[0]}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# Dashboard metrics row
# ---------------------------------------------------

recent_activity = st.session_state.get("recent_activity", [])
total_runs = sum(item["count"] for item in recent_activity)
last_run_time = recent_activity[0]["time"] if recent_activity else "—"

m1, m2, m3, m4 = st.columns(4)
m1.metric("API Status", status_badge[0].split(" ", 1)[1])
m2.metric("Model", "Random Forest")
m3.metric("Predictions This Session", total_runs)
m4.metric("Last Activity", last_run_time)

st.divider()

# ---------------------------------------------------
# Quick actions
# ---------------------------------------------------

col1, col2 = st.columns(2)
with col1:
    st.page_link(
        "pages/1_Single_Prediction.py",
        label="🚀 Start a Single Prediction",
        use_container_width=True,
    )
with col2:
    st.page_link(
        "pages/2_Batch_Prediction.py",
        label="📂 Run a Batch Prediction",
        use_container_width=True,
    )

st.divider()

# ---------------------------------------------------
# Why MachineGuard AI (replaces "Platform Features")
# ---------------------------------------------------

st.subheader("Why MachineGuard AI", anchor=False)

w1, w2, w3 = st.columns(3)

with w1:
    st.markdown(
        """
        <div class="mg-card">
            <h4>🛑 Prevent Downtime</h4>
            <p>Catch failure signals in sensor data early, so machines get
            serviced before they break down mid-shift.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with w2:
    st.markdown(
        """
        <div class="mg-card">
            <h4>💰 Cut Maintenance Cost</h4>
            <p>Move from fixed maintenance schedules to condition-based
            servicing, so you spend on repairs only when needed.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with w3:
    st.markdown(
        """
        <div class="mg-card">
            <h4>📈 Improve Reliability</h4>
            <p>Track risk levels over time per machine to spot patterns
            and plan capacity with confidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------
# How it Works
# ---------------------------------------------------

st.subheader("How It Works", anchor=False)

step1, step2, step3 = st.columns(3)

with step1:
    st.markdown(
        """
        <div class="mg-card">
            <h4>1️⃣ Enter Data</h4>
            <p>Provide machine sensor values directly, or upload a CSV
            for many machines at once.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with step2:
    st.markdown(
        """
        <div class="mg-card">
            <h4>2️⃣ AI Prediction</h4>
            <p>MachineGuard sends the request to the prediction API,
            backed by a trained Random Forest model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with step3:
    st.markdown(
        """
        <div class="mg-card">
            <h4>3️⃣ Results</h4>
            <p>Get back failure probability, risk level, and a clear
            prediction you can act on.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------
# Recent Activity
# ---------------------------------------------------

st.subheader("Recent Activity", anchor=False)

if recent_activity:
    for item in recent_activity:
        risk = item["risk_level"]
        risk_class = {
            "High": "mg-badge-red",
            "Medium": "mg-badge-yellow",
            "Low": "mg-badge-green",
        }.get(risk, "mg-badge-yellow")

        st.markdown(
            f"""
            <div class="mg-activity-row">
                <span>🕒 {item['time']} — {item['type']} ({item['count']} record{'s' if item['count'] != 1 else ''})</span>
                <span class="mg-badge {risk_class}">{risk}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info(
        "No predictions run yet this session. Try a **Single Prediction** "
        "or **Batch Prediction** above — activity will show up here."
    )

# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.divider()

st.caption(
    "MachineGuard AI • Production MLOps Demo • FastAPI • MLflow • Docker • Render • Streamlit"
)
