import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge # Switched to Ridge for tuning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from mlflow.models.signature import infer_signature

# Define the hyperparameters to test
ALPHA_VALUES = [0.1, 0.5, 1.0, 5.0, 10.0] 

def train_and_log_model(X_train, X_test, y_train, y_test, alpha):
    """Trains, evaluates, and logs a Ridge model for a given alpha."""
    
    with mlflow.start_run(run_name=f"Ridge_alpha_{alpha}") as run:
        # 1. Train Model
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        # 2. Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 3. Log Parameters and Metrics
        # Log the hyperparameter used
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("alpha", alpha)
        
        # Log the evaluation metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        print(f"Run Alpha={alpha}: MSE={mse:.4f}, R2={r2:.4f}")

        # 4. Log the Model (Only log the 'best' model structure or log all for comparison)
        # For simplicity, we log the model signature and an example for all runs
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Log the model with a unique name based on its alpha value
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"DiabetesRidgeModel_Alpha{alpha}",
            signature=signature,
            input_example=X_train.iloc[:1],
        )
        
        return mse, r2, run.info.run_id # Return metrics for finding the best run later


def main():
    # Load and split dataset
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name="target")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set up MLflow
    mlflow.set_experiment("DiabetesHyperparameterTuning")
    
    # Store results to determine the overall best model
    best_r2 = -np.inf
    best_run_id = None
    
    # --- Hyperparameter Tuning Loop ---
    print("\nStarting Hyperparameter Tuning...")
    for alpha in ALPHA_VALUES:
        mse, r2, run_id = train_and_log_model(X_train, X_test, y_train, y_test, alpha)
        
        # Determine the best model based on R2 score
        if r2 > best_r2:
            best_r2 = r2
            best_run_id = run_id
            
    print("\n" + "="*50)
    print(f"Tuning completed. Best R2 score: {best_r2:.4f}")
    print(f"The best model run ID is: {best_run_id}")
    print(f"Inspect results in the MLflow UI by running: mlflow ui")
    print("="*50)

if __name__ == "__main__":
    import numpy as np # Import numpy here for use in the script
    main()