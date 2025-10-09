import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# MLflow server URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 3, 5, 7],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

EXPERIMENT_NAME = "IrisClassification"
mlflow.set_experiment(EXPERIMENT_NAME)

# Log candidate runs
for params in ParameterGrid(param_grid):
    with mlflow.start_run():
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("accuracy", acc)

        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train[:1]

        mlflow.sklearn.log_model(
            sk_model=model,
            name="rf_model",
            signature=signature,
            input_example=input_example
        )

# Automatically register best run
client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1
)
best_run = runs[0]
best_run_id = best_run.info.run_id
best_acc = best_run.data.metrics["accuracy"]

print(f"Best run_id: {best_run_id} | Accuracy: {best_acc:.4f}")

model_uri = f"runs:/{best_run_id}/rf_model"
mlflow.register_model(model_uri, "IrisRFModel")
print("Best model registered successfully.")
