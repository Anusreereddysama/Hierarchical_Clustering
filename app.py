import streamlit as st
import numpy as np
import pickle
from sklearn.cluster import AgglomerativeClustering

# Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Hierarchical Clustering")

st.title("🌳 Customer Segmentation using Hierarchical Clustering")

st.write("Enter Customer Details")

income = st.number_input(
    "Annual Income",
    min_value=0.0,
    max_value=150.0
)

score = st.number_input(
    "Spending Score",
    min_value=0.0,
    max_value=100.0
)

if st.button("Find Cluster"):

    # Sample dataset
    data = np.array([
        [15,39],[16,81],[17,6],[18,77],
        [19,40],[20,76],[60,55],[62,59],
        [65,60],[68,63],[70,65],[72,68],
        [income,score]
    ])

    scaled_data = scaler.transform(data)

    model = AgglomerativeClustering(
        n_clusters=3,
        linkage='ward'
    )

    clusters = model.fit_predict(scaled_data)

    user_cluster = clusters[-1]

    st.success(
        f"Customer belongs to Cluster {user_cluster}"
    )