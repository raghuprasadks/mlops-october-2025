from flask import Flask, request, jsonify
import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris

app = Flask(__name__)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
MODEL_NAME = os.getenv("MODEL_NAME", "IrisRFModel")
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@latest")

iris = load_iris()
target_names = iris.target_names

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data["features"]
        prediction = model.predict([features])
        class_index = int(prediction[0])
        species = target_names[class_index]
        probabilities = model.predict_proba([features])[0]
        confidence = round(probabilities[class_index] * 100, 2)

        return jsonify({
            "class_index": class_index,
            "species": species,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
