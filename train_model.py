import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Sample customer dataset
data = {
    "Annual_Income": [15,16,17,18,19,20,60,62,65,68,70,72],
    "Spending_Score": [39,81,6,77,40,76,55,59,60,63,65,68]
}

df = pd.DataFrame(data)

X = df[['Annual_Income', 'Spending_Score']]

# Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Train Hierarchical Clustering
model = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

labels = model.fit_predict(X_scaled)

# Save scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Save labels
pickle.dump(labels, open("model.pkl", "wb"))

print("Hierarchical Clustering Model Saved!")