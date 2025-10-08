# 1-train.py
# pip install mlflow scikit-learn pandas

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlflow.models.signature import infer_signature

# Load Wine dataset
wine = load_wine(as_frame=True)
X = wine.data  # DataFrame with 13 features

# Train KMeans
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Evaluate clustering
sil_score = silhouette_score(X, labels)
print(f"Silhouette Score: {sil_score:.4f}")

# MLflow experiment
mlflow.set_experiment("WineClustering")

# input example and signature
input_example = X.iloc[:1]
signature = infer_signature(input_example, kmeans.predict(input_example))

with mlflow.start_run():
    mlflow.log_param("n_clusters", n_clusters)
    mlflow.log_metric("silhouette_score", sil_score)
    
    # Log & register model
    mlflow.sklearn.log_model(
        sk_model=kmeans,
        name="WineClusteringModel",
        registered_model_name="WineClustering",
        input_example=input_example,
        signature=signature
    )

print("MLflow run completed and model registered.")
