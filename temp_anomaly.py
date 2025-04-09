import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
xls = pd.ExcelFile(r"C:\Users\user\Downloads\real_time_weather_dataset_1000x1000.xlsx")
df_raw = xls.parse(xls.sheet_names[0])

# Select and clean temperature columns
temp_cols = [col for col in df_raw.columns if col.startswith("Temp")]
df_temp = df_raw[temp_cols].ffill(axis=1).bfill(axis=1)

# Compute rolling z-scores
window = 5
def rolling_z_scores(row):
    s = pd.Series(row)
    mean = s.rolling(window, min_periods=1, center=True).mean()
    std = s.rolling(window, min_periods=1, center=True).std()
    z = (s - mean) / (std + 1e-8)
    return z.values

z_scores_df = df_temp.apply(rolling_z_scores, axis=1, result_type='expand')

# Anomalies where |z| > 2.5
anomalies_df = z_scores_df.abs() > 2.5
df_temp["Anomaly_Count"] = anomalies_df.sum(axis=1)

# Top 5 rows with most anomalies
top_anomalies = df_temp.sort_values("Anomaly_Count", ascending=False).head(5)

# --- Visualization ---
for i, idx in enumerate(top_anomalies.index):
    temp_series = df_temp.loc[idx, temp_cols].values
    z_series = z_scores_df.loc[idx].values
    is_anomaly = np.abs(z_series) > 2.5

    plt.figure(figsize=(12, 4))
    plt.plot(range(1, 101), temp_series, label="Temperature", color="steelblue")
    plt.scatter(np.where(is_anomaly)[0]+1, temp_series[is_anomaly], color="red", label="Anomaly", zorder=5)
    plt.title(f"Temperature Anomaly Detection - Row {idx}")
    plt.xlabel("Time Step")
    plt.ylabel("Temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Heatmap of Z-scores for Top 5 Rows ---
plt.figure(figsize=(12, 5))
sns.heatmap(z_scores_df.loc[top_anomalies.index], cmap="coolwarm", center=0, cbar_kws={'label': 'Z-score'})
plt.title("Z-score Heatmap of Top 5 Anomalous Rows")
plt.xlabel("Time Step (Temp_1 to Temp_100)")
plt.ylabel("Top Rows")
plt.tight_layout()
plt.show()
