import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

# --- Configuration ---
# 1. REPLACE THIS WITH THE ACTUAL RUN ID OF YOUR BEST MODEL
# (e.g., the best_run_id determined after hyperparameter tuning)
#BEST_RUN_ID = "YOUR_BEST_MLFLOW_RUN_ID" 
BEST_RUN_ID = "e4359a7439ff4b759caf40865c3ead60"

# 2. Define the artifact path where the model was logged inside the run
# (This was 'model' in the previous tuning example)
MODEL_ARTIFACT_PATH = "model" 

# 3. Define the name under which the model was logged (used in the URI)
MODEL_NAME = "DiabetesRidgeModel_Alpha1.0" # Example model name

# --- MLflow Model Loading ---
# Construct the URI to load the model artifact
# Format: runs:/<run_id>/<artifact_path>
MODEL_URI = f"runs:/{BEST_RUN_ID}/{MODEL_ARTIFACT_PATH}"

# For a Registered Model, the URI would be:
# MODEL_URI = f"models/{MODEL_NAME}/1" # Example for Version 1

try:
    # Load the model outside the prediction function for efficiency
    loaded_model = mlflow.sklearn.load_model(MODEL_URI)
    print(f"✅ Successfully loaded model from URI: {MODEL_URI}")
except Exception as e:
    print(f"❌ Error loading model from MLflow URI: {MODEL_URI}")
    print(f"Error details: {e}")
    loaded_model = None # Set to None if loading fails

# --- Flask Application ---
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data["features"]  # Expecting list of 10 numbers
        prediction = loaded_model.predict([features])
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# --- Running the API ---
if __name__ == '__main__':
    # NOTE: Set the MLFLOW_TRACKING_URI environment variable before running this script 
    # if your server is not running locally.
    # Example: export MLFLOW_TRACKING_URI=http://<your-server-ip>:5000
    
    if loaded_model is not None:
        print("\nAPI is ready. Send POST request to http://127.0.0.1:8000/predict")
        app.run(host="0.0.0.0", port=8000, debug=True)
    else:
        print("\nAPI startup aborted due to model loading error.")