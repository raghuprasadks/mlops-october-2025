# 3-streamlit_app.py
# pip install streamlit requests pandas scikit-learn matplotlib

import streamlit as st
import requests
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

st.set_page_config(page_title="Wine Clustering Demo", layout="centered")
st.title("Wine Clustering Demo (KMeans)")

wine = load_wine(as_frame=True)
X_df = wine.data
feature_names = list(X_df.columns)
defaults = X_df.iloc[0].tolist()

st.subheader("Input Features")
cols = st.columns(4)
features = []
for i, name in enumerate(feature_names):
    with cols[i % 4]:
        val = st.number_input(name, value=float(defaults[i]), format="%.6f")
        features.append(val)

if st.button("Predict Cluster"):
    try:
        resp = requests.post("http://127.0.0.1:8000/predict", json={"features": features}, timeout=10)
        if resp.status_code == 200:
            cluster = resp.json().get("cluster")
            st.success(f"Predicted cluster: {cluster}")
        else:
            st.error(f"API error ({resp.status_code}): {resp.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")

# Visualize clusters (fit locally for demo)
st.subheader("Wine dataset clusters visualized (PCA to 2D)")
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_df)
labels = kmeans.labels_
pca = PCA(n_components=2, random_state=42)
X2 = pca.fit_transform(X_df)

plot_df = pd.DataFrame({
    "PC1": X2[:,0],
    "PC2": X2[:,1],
    "cluster": labels
})

fig, ax = plt.subplots()
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for c in np.sort(plot_df['cluster'].unique()):
    subset = plot_df[plot_df['cluster'] == c]
    ax.scatter(subset['PC1'], subset['PC2'], label=f"Cluster {c}", s=40, alpha=0.7)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Wine clusters (KMeans) visualized with PCA")
ax.legend()
st.pyplot(fig)

st.subheader("Sample points and cluster labels")
st.dataframe(plot_df.head(8))
