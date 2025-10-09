import streamlit as st
import requests
import os

st.set_page_config(page_title="Iris Classifier", layout="centered")
st.title("🌼 Iris Flower Classification (MLflow + RandomForest)")

API_URL = os.getenv("API_URL", "http://localhost:8000")

features_names = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

default_values = [5.1, 3.5, 1.4, 0.2]
features = [st.number_input(name, value=default_values[i], format="%.2f") for i, name in enumerate(features_names)]

if st.button("Predict"):
    try:
        response = requests.post(f"{API_URL}/predict", json={"features": features})
        if response.status_code == 200:
            data = response.json()
            species = data["species"]
            confidence = data.get("confidence", None)
            if confidence:
                st.success(f"🌸 Predicted Species: **{species.capitalize()}** ({confidence}% confidence)")
            else:
                st.success(f"🌸 Predicted Species: **{species.capitalize()}**")
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
