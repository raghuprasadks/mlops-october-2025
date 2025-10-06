import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Load dataset
#boston = load_boston()
boston = fetch_california_housing()
X = boston.data
y = boston.target
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Display prediction for first record of X_test
print(f"Prediction for first record of X_test: {y_pred[0]:.2f}")
print(f"Actual value: {y_test[0]:.2f}")
