import streamlit as st

st.title("📖 About MachineGuard AI")

st.write(
"""
MachineGuard AI is an end-to-end Industrial Predictive
Maintenance platform.

The project demonstrates modern MLOps practices including:

• FastAPI Deployment

• Docker

• MLflow Model Registry

• Streamlit Frontend

• Render Deployment

• Monitoring

• CI/CD

• Machine Learning Pipeline
"""
)

st.divider()

st.subheader("Architecture")

st.code(
"""
Machine

   │

   ▼

Streamlit UI

   │ HTTPS

   ▼

FastAPI API (Render)

   │

Prediction Service

   │

Joblib Pipeline

   │

Prediction
"""
)