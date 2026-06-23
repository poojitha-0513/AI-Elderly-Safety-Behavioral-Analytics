import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Elderly Safety AI",
    layout="wide"
)

st.title(
    "AI-Powered Elderly Safety Monitoring System"
)

st.write(
    "Real-Time Elderly Safety and Behavioral Analytics"
)

model = joblib.load(
    "models/risk_model.pkl"
)

features = pd.read_csv(
    "data/features.csv"
)

risk_prediction = model.predict(
    features
)

risk_level = risk_prediction[0]

st.header("Current Risk Level")

if risk_level == "Low":

    st.success(
        f"Risk Level: {risk_level}"
    )

elif risk_level == "Medium":

    st.warning(
        f"Risk Level: {risk_level}"
    )

else:

    st.error(
        f"Risk Level: {risk_level}"
    )

st.header(
    "Behavior Feature Summary"
)

st.dataframe(
    features
)

st.header(
    "Activity Statistics"
)

activity_columns = [

    "standing_time",
    "walking_time",
    "sitting_time",
    "lying_time"

]

chart_data = features[
    activity_columns
].T

chart_data.columns = [
    "Count"
]

st.bar_chart(
    chart_data
)

st.header(
    "Recent Alerts"
)

if os.path.exists(
    "data/anomaly_logs.csv"
):

    alerts = pd.read_csv(
        "data/anomaly_logs.csv"
    )

    st.dataframe(
        alerts.tail(10)
    )

else:

    st.info(
        "No alerts found."
    )

st.header(
    "Privacy Protection"
)

st.success(
    """
    Video is processed locally.

    No face images are stored.

    No video recordings are saved.

    Only skeletal movement data and activity statistics are retained.
    """
)

st.markdown("---")

st.write(
    "Elderly Care Monitoring System"
)