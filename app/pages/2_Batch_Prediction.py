"""Batch Prediction."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.api import batch_predict

st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Batch Prediction")

st.caption(
    "Upload a CSV containing multiple machine records."
)

st.info(
    """
Required columns:

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

if uploaded_file:

    dataframe = pd.read_csv(
        uploaded_file,
    )

    st.subheader("Preview")

    st.dataframe(
        dataframe,
        use_container_width=True,
    )

    if st.button(
        "🚀 Run Batch Prediction",
        use_container_width=True,
    ):

        with st.spinner(
            "Processing..."
        ):

            result = batch_predict(
                dataframe.to_dict(
                    orient="records",
                )
            )

        if result is None:

            st.error(
                "Batch prediction failed."
            )

        else:

            output = pd.DataFrame(
                result
            )

            st.success(
                f"{len(output)} predictions completed."
            )

            st.subheader(
                "Prediction Results"
            )

            st.dataframe(
                output,
                use_container_width=True,
            )

            csv = output.to_csv(
                index=False,
            ).encode()

            st.download_button(
                "⬇ Download Predictions",
                csv,
                "predictions.csv",
                "text/csv",
                use_container_width=True,
            )