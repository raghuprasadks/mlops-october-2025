import streamlit as st
import requests
import numpy as np

st.title("Iris Flower Classification (API Demo)")

st.header("Enter Iris Flower Features:")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict"):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    url = "http://127.0.0.1:5000/predict"
    response = requests.post(url, json={"features": features})
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        species = ["Setosa", "Versicolor", "Virginica"]
        st.success(f"Predicted Species: {species[prediction]}")
    else:
        st.error("API request failed. Is the Flask server running?")