import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from mlflow.models.signature import infer_signature

def main():
    # Load dataset
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name="target")

    # Preview dataset
    print("Dataset preview:")
    print(pd.concat([X, y], axis=1).head())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start MLflow experiment
    mlflow.set_experiment("DiabetesLinearRegression")

    with mlflow.start_run() as run:
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        print(f"Model training completed.\nMSE: {mse:.4f}, R2: {r2:.4f}")

        # Infer model signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log model with new param `name`
        mlflow.sklearn.log_model(
            model,
            name="DiabetesLinearRegression",
            registered_model_name="DiabetesLinearRegression",
            input_example=X_train.iloc[:1],   # example row
            signature=signature
        )

        print(f"✅ Run completed. Run ID: {run.info.run_id}")
        print("✅ Model registered as: DiabetesLinearRegression")

if __name__ == "__main__":
    main()
