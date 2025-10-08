# 1-train.py
# Simple classification with MLflow (Iris dataset)

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# -----------------------
# Load dataset
# -----------------------
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# Train model
# -----------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# -----------------------
# MLflow experiment
# -----------------------
mlflow.set_experiment("IrisClassification")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    
    # Log metrics
    mlflow.log_metric("accuracy", acc)
    
    # Log AND register model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="rf_classifier",
        registered_model_name="IrisRFModel"  # This will appear in the Model Registry
    )

print("MLflow run completed and model logged & registered.")
