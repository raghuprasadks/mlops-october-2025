from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Test prediction for 1st and 3rd dataset records
first_pred = model.predict([X_test[0]])
third_pred = model.predict([X_test[2]])
print(f"Prediction for 1st test record: {first_pred[0]}, Actual: {y_test[0]}")
print(f"Prediction for 3rd test record: {third_pred[0]}, Actual: {y_test[2]}")

# Save the trained model to a pickle file
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)
