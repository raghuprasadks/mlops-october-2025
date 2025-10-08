# 2-app.py
# pip install flask mlflow scikit-learn pandas

from flask import Flask, request, jsonify
import mlflow.sklearn
import traceback

app = Flask(__name__)

MODEL_NAME = "WineClustering"

# Load latest model
try:
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@latest")
    print(f"Loaded MLflow model: models:/{MODEL_NAME}@latest")
except Exception as e:
    print("Failed to load model from MLflow registry.")
    traceback.print_exc()
    model = None

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "JSON must contain 'features'"}), 400
        features = data["features"]
        if not isinstance(features, (list, tuple)) or len(features) != 13:
            return jsonify({"error": "Expected list of 13 features"}), 400

        cluster = model.predict([features])[0]
        return jsonify({"cluster": int(cluster)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=8000, debug=True)
