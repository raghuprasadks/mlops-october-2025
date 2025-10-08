# streamlit_app.py
# pip install streamlit requests

import streamlit as st
import requests

st.set_page_config(page_title="Diabetes Progression Predictor", layout="centered")
st.title("Diabetes Progression Prediction (MLflow + Linear Regression)")

st.write("Enter the values for each feature:")

# Feature names from the sklearn diabetes dataset
feature_names = [
    "age", "sex", "bmi", "bp",
    "s1", "s2", "s3", "s4", "s5", "s6"
]

# Default sample values from the dataset
default_values = [
    0.038076, 0.050680, 0.061696, 0.021872,
    -0.044223, -0.034821, -0.043401, -0.002592,
    0.019907, -0.017646
]

# Collect feature inputs from user with defaults
features = []
for i, name in enumerate(feature_names):
    val = st.number_input(f"{name}", value=default_values[i], format="%.6f")
    features.append(val)

# Predict button
if st.button("Predict"):
    try:
        # Send request to Flask API
        response = requests.post(
            "http://127.0.0.1:8000/predict",  # Make sure your Flask API is running on this port
            json={"features": features}
        )
        if response.status_code == 200:
            prediction = response.json()["prediction"][0]
            st.success(f"Predicted Diabetes Progression: {prediction:.2f}")
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
