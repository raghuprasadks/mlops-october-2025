import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient # Needed for programmatic promotion
import pandas as pd
import numpy as np

# --- Configuration ---
ALPHA_VALUES = [0.1, 0.5, 1.0, 5.0, 10.0]
MODEL_REGISTRY_NAME = "DiabetesRidgeModel-new"
EXPERIMENT_NAME = "DiabetesHyperparameterTuning-new"

def train_and_log_model(X_train, X_test, y_train, y_test, alpha, registry_name):
    """Trains, evaluates, and logs a Ridge model, registering it."""
    
    with mlflow.start_run(run_name=f"Ridge_alpha_{alpha}") as run:
        # 1. Train Model
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        # 2. Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 3. Log Parameters and Metrics
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        print(f"Run Alpha={alpha}: R2={r2:.4f}, Logged as version...")

        # 4. Log and Register the Model
        signature = infer_signature(X_train, model.predict(X_train))
        
        log_info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=registry_name,
            signature=signature,
            input_example=X_train.iloc[:1],
        )
        
        # Return key identifiers for tracking the best model
        return r2, log_info.version

def promote_best_model(client, registry_name, best_version):
    """Archives old production models and promotes the new best version."""
    
    # 1. Archive any existing model in 'Production'
    for mv in client.get_latest_versions(registry_name, stages=["Production"]):
        if mv.version != best_version: # Only archive if it's not the new best
            client.transition_model_version_stage(
                name=registry_name,
                version=mv.version,
                stage="Archived"
            )
            print(f"-> Archived old Production version {mv.version}.")

    # 2. Transition the new best version to 'Production'
    client.transition_model_version_stage(
        name=registry_name,
        version=best_version,
        stage="Production"
    )
    print(f"-> ✅ Version {best_version} transitioned to **Production**.")


def main():
    # --- Data Setup ---
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name="target")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- MLflow Setup ---
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    # Variables to track the overall best model across all runs
    best_r2 = -np.inf
    best_version = None
    
    print(f"Starting Hyperparameter Tuning for {MODEL_REGISTRY_NAME}...")
    
    # --- Hyperparameter Tuning Loop ---
    for alpha in ALPHA_VALUES:
        r2, version = train_and_log_model(X_train, X_test, y_train, y_test, alpha, MODEL_REGISTRY_NAME)
        
        # Track the best model version
        if r2 > best_r2:
            best_r2 = r2
            best_version = version # Store the version number of the best model
    
    # --- Promotion ---
    if best_version:
        print("\n" + "="*70)
        print(f"TUNING COMPLETE. Best R2: {best_r2:.4f} (Version {best_version})")
        promote_best_model(client, MODEL_REGISTRY_NAME, best_version)
        print("="*70)
        print(f"To use in the Flask API, load model from URI: models:/{MODEL_REGISTRY_NAME}/Production")
    else:
        print("No models were successfully trained or identified as the best.")


if __name__ == "__main__":
    main()