import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load dataset
xls = pd.ExcelFile(r"C:\Users\user\Downloads\real_time_weather_dataset_1000x1000.xlsx")
df_raw = xls.parse(xls.sheet_names[0])

# Identify relevant columns
temp_cols = [col for col in df_raw.columns if col.startswith("Temp")]
desc_cols = [col for col in df_raw.columns if col.startswith("Description")]

# Handle missing values in temperature and descriptions
df_desc = df_raw[desc_cols].fillna("").astype(str).apply(lambda col: col.str.strip().str.lower())
df_temp = df_raw[temp_cols].ffill(axis=1).bfill(axis=1)

# --- Encode common weather descriptions ---
unique_descriptions = pd.Series(df_desc.values.ravel()).value_counts().head(10).index.tolist()
desc_to_id = {desc: i for i, desc in enumerate(unique_descriptions)}
id_to_desc = {i: desc for desc, i in desc_to_id.items()}

# Filter descriptions to top known types only
df_desc_encoded = df_desc.apply(lambda col: col.map(lambda x: desc_to_id.get(x, -1)))

# --- Forecasting Logic ---
def forecast_next_description(row_temp, row_desc):
    temp_trend = np.sign(np.diff(row_temp[-6:]))  # last 5 changes
    last_desc = row_desc[-1]

    if last_desc == desc_to_id.get("cloudy") and np.sum(temp_trend) > 2:
        return "sunny"
    elif last_desc == desc_to_id.get("sunny") and np.sum(temp_trend) < -2:
        return "rainy"
    elif last_desc == desc_to_id.get("rainy") and np.sum(temp_trend) > 0:
        return "cloudy"
    else:
        return id_to_desc.get(last_desc, "unknown")

df_temp_np = df_temp[temp_cols].values
df_desc_np = df_desc_encoded[desc_cols].values

forecasted_desc = []
for i in range(df_temp_np.shape[0]):
    pred = forecast_next_description(df_temp_np[i], df_desc_np[i])
    forecasted_desc.append(pred)

# Add forecast results
df_result = df_raw.copy()
df_result["Forecast_Description_101"] = forecasted_desc

# Display sample
print("\nSample forecasted descriptions:")
print(df_result[["Forecast_Description_101"]].head())

# --- Visualization: Forecast distribution ---
# Plot forecasted description counts
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df_result,
    y="Forecast_Description_101",
    hue="Forecast_Description_101",  # assign hue same as y
    palette="viridis",
    legend=False
)
plt.title("Forecasted Weather Descriptions", fontsize=14)
plt.xlabel("Count")
plt.ylabel("Description")
plt.tight_layout()
plt.show()


### --- Visualization: Temperature trends that led to 'sunny' forecast ---
##
### Filter rows for a specific forecast, e.g., 'sunny'
##selected_desc = "sunny"  # or "mist", "haze", etc.
##selected_rows = df_result[df_result["Forecast_Description_101"] == selected_desc].copy()
##
##plt.figure(figsize=(12, 6))
##colors = sns.color_palette("Set2", len(selected_rows))
##
##for idx, (i, row) in enumerate(selected_rows.iterrows()):
##    temps = row[temp_cols].astype(float)
##    smoothed = smooth(temps)
##    plt.plot(smoothed, label=f"Row {i}", color=colors[idx], linewidth=2)
##
### Only show legend if there's data
##if not selected_rows.empty:
##    plt.legend(title="Row Index", loc="upper left", bbox_to_anchor=(1, 1))
##
##plt.title("Smoothed Temperature Trends Leading to 'Sunny' Forecast", fontsize=16)
##plt.xlabel("Time Step", fontsize=12)
##plt.ylabel("Temperature (°C)", fontsize=12)
##plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
##plt.tight_layout()
##plt.show()

