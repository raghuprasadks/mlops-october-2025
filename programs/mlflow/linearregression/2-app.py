# app.py
# pip install flask mlflow scikit-learn

from flask import Flask, request, jsonify
import mlflow.sklearn

app = Flask(__name__)

# Load latest registered model
MODEL_NAME = "DiabetesLinearRegression"
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data["features"]  # Expecting list of 10 numbers
        prediction = model.predict([features])
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(port=8000, debug=True)
