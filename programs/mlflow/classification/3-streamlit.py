# streamlit_classification.py

import streamlit as st
import requests

st.set_page_config(page_title="Iris Classifier", layout="centered")
st.title("Iris Flower Classification (MLflow + RandomForest)")

st.write("Enter the values for each feature:")

# Feature names
features_names = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

# Default sample values (from first row of iris dataset)
default_values = [5.1, 3.5, 1.4, 0.2]

features = []
for i, name in enumerate(features_names):
    val = st.number_input(name, value=default_values[i])
    features.append(val)

if st.button("Predict"):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"features": features}
        )
        if response.status_code == 200:
            pred_class = response.json()["prediction"]
            st.success(f"Predicted Iris Class: {pred_class}")
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
