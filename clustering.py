import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 📥 Load data
file_path = r"C:\Users\user\Downloads\real_time_weather_dataset_1000x1000.xlsx"
df = pd.read_excel(file_path)

# 🎯 Extract temperature and description columns
temp_cols = [col for col in df.columns if 'Temp_' in col]
desc_cols = [col for col in df.columns if 'Description_' in col]

# 🧮 Compute average temperature per record
df['Avg_Temp'] = df[temp_cols].mean(axis=1)

# 🧠 One-hot encode weather descriptions per row
desc_freq = df[desc_cols].apply(lambda row: row.value_counts(), axis=1).fillna(0)
desc_freq = desc_freq.astype(int)

# 🌡️ Combine temp and description frequencies into a feature matrix
features = pd.concat([df['Avg_Temp'], desc_freq], axis=1)

# 🎨 Normalize features
features_norm = (features - features.mean()) / features.std()

# 🧪 Manual K-Means implementation
def kmeans(X, k=3, max_iter=100, seed=42):
    np.random.seed(seed)
    centroids = X.sample(n=k).to_numpy()
    
    for _ in range(max_iter):
        # Assign clusters
        distances = np.linalg.norm(X.to_numpy()[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([X[labels == i].mean().to_numpy() for i in range(k)])
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids
    
    return labels, centroids

# 🔍 Perform clustering
labels, centroids = kmeans(features_norm, k=4)
features['Cluster'] = labels

# 📊 Visualization: Clustered Average Temperature vs Most Frequent Description
features['Dominant_Description'] = desc_freq.idxmax(axis=1)
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=features,
    x='Avg_Temp',
    y='Dominant_Description',
    hue='Cluster',
    palette='tab10',
    s=80,
    edgecolor='black'
)
plt.title("Weather Profile Clustering", fontsize=16)
plt.xlabel("Average Temperature (°C)")
plt.ylabel("Most Frequent Description")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 📝 Optional: Cluster Summary
cluster_summary = features.groupby('Cluster')['Avg_Temp'].agg(['mean', 'min', 'max'])
print("\nCluster Temperature Summary:")
print(cluster_summary)
