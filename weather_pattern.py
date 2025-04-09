import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
xls = pd.ExcelFile(r"C:\Users\user\Downloads\real_time_weather_dataset_1000x1000.xlsx")
df_raw = xls.parse(xls.sheet_names[0])

# Identify temp & description columns
temp_cols = [col for col in df_raw.columns if col.startswith("Temp")]
desc_cols = [col for col in df_raw.columns if col.startswith("Description")]

# Fill missing temperature values
df_temp = df_raw[temp_cols].fillna(df_raw[temp_cols].mean())

# Normalize description text (apply + map version)
desc_data = df_raw[desc_cols].fillna("").apply(lambda col: col.map(lambda x: str(x).lower().strip()))

# Extract most frequent description per row
def most_frequent(row):
    non_empty = row[row != ""]
    return non_empty.mode().iloc[0] if not non_empty.mode().empty else "unknown"

labels = desc_data.apply(most_frequent, axis=1)

# Encode labels manually
label_map = {label: idx for idx, label in enumerate(labels.unique())}
inv_label_map = {v: k for k, v in label_map.items()}
encoded_labels = labels.map(label_map)

# Combine into a clean DataFrame
df = pd.concat([df_temp.reset_index(drop=True), labels.rename("Label"), encoded_labels.rename("Encoded_Label")], axis=1)

# Features and labels
X = df[temp_cols].to_numpy()
y = df["Encoded_Label"].to_numpy()

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Nearest Centroid Classifier
centroids = {label: X_train[y_train == label].mean(axis=0) for label in np.unique(y_train)}

def predict(X):
    predictions = []
    for x in X:
        distances = {label: np.linalg.norm(x - centroid) for label, centroid in centroids.items()}
        predictions.append(min(distances, key=distances.get))
    return np.array(predictions)

# Predict and evaluate
y_pred = predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
conf_matrix = pd.crosstab(pd.Series(y_test, name="Actual"),
                          pd.Series(y_pred, name="Predicted"))

conf_matrix.index = conf_matrix.index.map(inv_label_map)
conf_matrix.columns = conf_matrix.columns.map(inv_label_map)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Weather Classification")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
