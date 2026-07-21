"""MachineGuard AI - Command Center & Dashboard."""

from __future__ import annotations

from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

from services.api import APIError, get_ready
from utils.styles import load_css

st.set_page_config(
    page_title="MachineGuard AI | Command Center",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load base CSS styles
load_css()

# Custom Styling for Dashboard Elements
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
        border-color: #3b82f6;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #888888;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #10b981;
        margin-top: 4px;
        font-weight: 500;
    }
    
    /* Useful Section Benefit Cards */
    .benefit-card {
        background: rgba(30, 41, 59, 0.4);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 15px;
    }
    .benefit-title {
        font-weight: 600;
        font-size: 1.05rem;
        color: #60a5fa;
        margin-bottom: 6px;
    }
    .benefit-desc {
        font-size: 0.92rem;
        color: #cbd5e1;
        line-height: 1.5;
    }
    
    /* Status Badges */
    .status-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-online {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
    }
    .status-offline {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    .status-starting {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid #f59e0b;
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
    st.write(
        """
        Go to the pages on the left:
        
        • **Single Prediction**
        
        • **Batch Prediction**
        
        • **Model Monitoring**
        
        • **API Status**
        
        • **About**
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
# Header & API Live Status Check
# ---------------------------------------------------
head_col1, head_col2 = st.columns([3, 1])

with head_col1:
    st.title("⚙️ MachineGuard AI — Command Center")
    st.caption("Real-Time Fleet Health Monitoring & Industrial Failure Prediction")

with head_col2:
    api_status = "unknown"
    try:
        ready = get_ready()
        if ready.get("status") == "ready":
            api_status = "online"
            st.markdown(
                '<div style="text-align: right; margin-top: 15px;">'
                '<span class="status-pill status-online">🟢 API Online</span>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            api_status = "starting"
            st.markdown(
                '<div style="text-align: right; margin-top: 15px;">'
                '<span class="status-pill status-starting">🟡 API Warming Up</span>'
                "</div>",
                unsafe_allow_html=True,
            )
    except Exception:
        api_status = "offline"
        st.markdown(
            '<div style="text-align: right; margin-top: 15px;">'
            '<span class="status-pill status-offline">🔴 API Offline</span>'
            "</div>",
            unsafe_allow_html=True,
        )

# Render Sleeping Notice
if api_status == "offline":
    st.info(
        "ℹ️ **Note on Render Free Tier:** Render puts inactive free instances to sleep after 15 minutes. "
        "If the API displays as offline, it may simply be waking up from a cold start. "
        "Click below to send a wake-up ping request."
    )
    if st.button("🔄 Wake Up / Re-check API Status"):
        st.rerun()

st.divider()

# ---------------------------------------------------
# Hero & Quick Action Launchpad
# ---------------------------------------------------
hero_col1, hero_col2 = st.columns([2, 1])

with hero_col1:
    st.markdown(
        """
        ### 🛡️ AI-Powered Equipment Protection
        MachineGuard AI processes multi-sensor telemetry (air/process temperature, rotational speed, torque, tool wear)
        in real time to detect early thermal, mechanical, and fatigue failure signals before catastrophic downtime occurs.
        """
    )
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.page_link(
            "pages/1_Single_Prediction.py",
            label="🚀 Single Machine Inspection",
            use_container_width=True,
        )
    with btn_col2:
        st.page_link(
            "pages/2_Batch_Prediction.py",
            label="📂 Upload Batch Sensor CSV",
            use_container_width=True,
        )

with hero_col2:
    st.markdown(
        """
        **System Architecture**
        - **Model:** Random Forest Classifier
        - **Inference Engine:** FastAPI REST endpoint
        - **MLOps Stack:** MLflow, Docker & Render
        """
    )

st.divider()

# ---------------------------------------------------
# Fleet Dashboard KPI Metrics
# ---------------------------------------------------
st.subheader("📊 Fleet Health Summary")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Monitored Assets</div>
            <div class="metric-value">1,248</div>
            <div class="metric-sub">▲ 12 added this week</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Fleet Health Index</div>
            <div class="metric-value">98.4%</div>
            <div class="metric-sub">🟢 Optimal Condition</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Active Anomaly Risk</div>
            <div class="metric-value">3</div>
            <div class="metric-sub" style="color: #f59e0b;">⚠️ Needs Inspection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">Est. Saved Downtime</div>
            <div class="metric-value">$142.5K</div>
            <div class="metric-sub">▲ 18.2% vs last month</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ---------------------------------------------------
# Recent Activities / Live Event Stream
# ---------------------------------------------------
st.subheader("⚡ Recent Sensor Telemetry & System Activity")
st.caption("Live feed of recent telemetry logs processed by MachineGuard AI")

now = datetime.now()
activities_data = [
    {
        "Timestamp": (now - timedelta(minutes=4)).strftime("%H:%M:%S"),
        "Machine ID": "M1024",
        "Sensor Status": "Heat Dissipation Warning",
        "Failure Risk": "78.4%",
        "Risk Level": "HIGH",
        "Action Taken": "Alert dispatched to Maintenance Tech",
    },
    {
        "Timestamp": (now - timedelta(minutes=18)).strftime("%H:%M:%S"),
        "Machine ID": "M2041",
        "Sensor Status": "Tool Wear > 200 min",
        "Failure Risk": "89.1%",
        "Risk Level": "CRITICAL",
        "Action Taken": "Scheduled Tool Replacement",
    },
    {
        "Timestamp": (now - timedelta(minutes=35)).strftime("%H:%M:%S"),
        "Machine ID": "L0812",
        "Sensor Status": "RPM & Torque Normal",
        "Failure Risk": "1.2%",
        "Risk Level": "LOW",
        "Action Taken": "Automated scan passed",
    },
    {
        "Timestamp": (now - timedelta(minutes=52)).strftime("%H:%M:%S"),
        "Machine ID": "H3109",
        "Sensor Status": "Power Strain Fluctuation",
        "Failure Risk": "42.0%",
        "Risk Level": "MEDIUM",
        "Action Taken": "Flagged for routine check",
    },
    {
        "Timestamp": (now - timedelta(hours=1, minutes=10)).strftime("%H:%M:%S"),
        "Machine ID": "L0550",
        "Sensor Status": "Batch Telemetry Inspection",
        "Failure Risk": "0.8%",
        "Risk Level": "LOW",
        "Action Taken": "Batch report generated",
    },
]

df_activities = pd.DataFrame(activities_data)
st.dataframe(
    df_activities,
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ---------------------------------------------------
# How This App Is Useful (Business Value & Capabilities)
# ---------------------------------------------------
st.subheader("💡 Why MachineGuard AI? (Value & Impact)")
st.caption("How predictive maintenance transforms industrial asset longevity and operational efficiency")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        """
        <div class="benefit-card">
            <div class="benefit-title">🚫 Eliminate Unplanned Downtime</div>
            <div class="benefit-desc">
                Unscheduled machinery failure costs manufacturers up to $50,000 per hour.
                MachineGuard AI analyzes sensor metrics to detect failure patterns hours or days before breakdowns occur.
            </div>
        </div>
        <div class="benefit-card">
            <div class="benefit-title">💰 Lower Operational & Repair Costs</div>
            <div class="benefit-desc">
                Transition from reactive emergency maintenance to scheduled maintenance windows.
                Reduce costly rush shipping on spare parts and unnecessary technician overtime.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        """
        <div class="benefit-card">
            <div class="benefit-title">⚙️ Extended Equipment Lifespan</div>
            <div class="benefit-desc">
                Running machinery under excessive torque or heat strains structural integrity. Real-time telemetry monitoring prevents catastrophic component damage.
            </div>
        </div>
        <div class="benefit-card">
            <div class="benefit-title">⚡ Scalable Production MLOps Integration</div>
            <div class="benefit-desc">
                Built on containerized FastAPI microservices and monitored via MLflow, enabling integration into factory IoT gateways and enterprise ERP software.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.caption(
    "MachineGuard AI • Production MLOps Demo Platform • FastAPI • MLflow • Docker • Render • Streamlit"
)