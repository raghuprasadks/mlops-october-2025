# app_classification.py

from flask import Flask, request, jsonify
import mlflow.sklearn

app = Flask(__name__)

# Load latest registered model
MODEL_NAME = "IrisRFModel"
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@latest")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data["features"]  # Expecting list of 4 numbers
        prediction = model.predict([features])
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(port=8000, debug=True)
