"""Batch prediction page."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from app.services.api import predict_batch
from app.utils.styles import render_prediction_card

st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📂",
    layout="wide",
)

st.title("📂 Batch Machine Failure Prediction")

st.markdown(
    """
Upload a CSV file containing multiple machine records and receive
failure predictions for every machine.
"""
)

st.info(
    """
Required CSV columns:

- machine_type
- air_temperature
- process_temperature
- rotational_speed
- torque
- tool_wear
"""
)

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"],
)

example = pd.DataFrame(
    [
        {
            "machine_type": "M",
            "air_temperature": 298.1,
            "process_temperature": 308.6,
            "rotational_speed": 1551,
            "torque": 42.8,
            "tool_wear": 0,
        },
        {
            "machine_type": "H",
            "air_temperature": 300.2,
            "process_temperature": 311.8,
            "rotational_speed": 1450,
            "torque": 51.2,
            "tool_wear": 120,
        },
    ]
)

buffer = io.StringIO()
example.to_csv(buffer, index=False)

st.download_button(
    "⬇ Download Sample CSV",
    buffer.getvalue(),
    file_name="machineguard_sample.csv",
    mime="text/csv",
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Preview")
        st.dataframe(
            df,
            use_container_width=True,
        )

        if st.button(
            "🚀 Predict Batch",
            use_container_width=True,
        ):

            with st.spinner("Predicting..."):

                result = predict_batch(df)

            predictions = pd.DataFrame(result)

            st.success(
                f"Processed {len(predictions)} machines successfully."
            )

            st.dataframe(
                predictions,
                use_container_width=True,
            )

            if "risk_level" in predictions.columns:

                st.subheader("Risk Distribution")

                st.bar_chart(
                    predictions["risk_level"].value_counts()
                )

            csv = predictions.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇ Download Predictions",
                csv,
                file_name="batch_predictions.csv",
                mime="text/csv",
            )

            if (
                "failure_probability" in predictions.columns
                and len(predictions) > 0
            ):

                st.subheader("First Prediction")

                render_prediction_card(
                    float(
                        predictions.loc[
                            0,
                            "failure_probability",
                        ]
                    ),
                    str(
                        predictions.loc[
                            0,
                            "risk_level",
                        ]
                    ),
                )

    except Exception as error:
        st.error(str(error))