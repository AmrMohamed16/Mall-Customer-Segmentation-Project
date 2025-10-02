import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛍️ Customer Segmentation App")

# Upload or load dataset
uploaded_file = st.file_uploader("Upload Mall Customer Dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.info("Using default Mall_Customers.csv dataset")
    data = pd.read_csv("Mall_Customers.csv")

st.write("### Dataset Preview")
st.dataframe(data.head())

# Fixed features
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = data[features]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose Algorithm
st.sidebar.header("Clustering Settings")
algo = st.sidebar.selectbox("Choose Clustering Algorithm", ["KMeans", "DBSCAN"])

if algo == "KMeans":
    k = st.sidebar.slider("Number of clusters (K)", 2, 10, 5)
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)
    data['Cluster'] = labels

    # Silhouette Score
    score = silhouette_score(X_scaled, labels)
    st.write(f"**Silhouette Score:** {score:.3f}")

elif algo == "DBSCAN":
    eps = st.sidebar.slider("EPS (Neighborhood size)", 0.1, 5.0, 0.5, 0.1)
    min_samples = st.sidebar.slider("Minimum Samples", 2, 20, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    data['Cluster'] = labels

    if len(set(labels)) > 1 and -1 not in labels:
        score = silhouette_score(X_scaled, labels)
        st.write(f"**Silhouette Score:** {score:.3f}")
    else:
        st.warning("DBSCAN could not form valid clusters. Try adjusting parameters.")

# Cluster Visualization
st.write("### Cluster Visualization")
fig, ax = plt.subplots()
sns.scatterplot(x=X[features[0]], y=X[features[1]], hue=data['Cluster'], palette="tab10", ax=ax)
st.pyplot(fig)

# Average spending per cluster
st.write("### Average Spending Score per Cluster")
avg_spending = data.groupby("Cluster")["Spending Score (1-100)"].mean()
st.bar_chart(avg_spending)

# Download clustered data
st.write("### Download Segmented Data")
csv = data.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV with Clusters", csv, "segmented_customers.csv", "text/csv")

# Prediction for new customers (only works for KMeans)
if algo == "KMeans":
    st.write("### Predict Cluster for New Customer")
    col1, col2 = st.columns(2)
    income = col1.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
    spending = col2.number_input("Spending Score (1-100)", min_value=0, max_value=100, value=50)
    
    new_data = scaler.transform([[income, spending]])
    cluster_pred = model.predict(new_data)[0]
    st.success(f"Predicted Cluster: {cluster_pred}")
