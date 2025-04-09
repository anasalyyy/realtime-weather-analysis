import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Optional: use a font that supports emojis
plt.rcParams['font.family'] = 'DejaVu Sans'

# 📥 Load the dataset
file_path =r"C:\Users\user\Downloads\real_time_weather_dataset_1000x1000.xlsx"
df = pd.read_excel(file_path)

# 🔍 Filter relevant columns
temp_columns = [col for col in df.columns if col.startswith("Temp_")]
desc_columns = [col for col in df.columns if col.startswith("Description_")]

# 🔄 Convert wide format to long format
temps_long = df[temp_columns].melt(var_name="Temp_Slot", value_name="Temperature")
descs_long = df[desc_columns].melt(var_name="Desc_Slot", value_name="Description")

# 🔗 Combine both into a single DataFrame
combined = pd.DataFrame({
    "Temperature": temps_long["Temperature"],
    "Description": descs_long["Description"]
})

# 🧹 Drop missing values
combined.dropna(inplace=True)

# 📊 Compute mean and variance by description
summary_stats = combined.groupby("Description")["Temperature"].agg(['mean', 'var']).reset_index()
summary_stats = summary_stats.sort_values("mean", ascending=False)

# 🎨 Set Seaborn theme
sns.set(style="whitegrid", font_scale=1.1)

# ────────────────────────────────────────────────
# 📈 Barplot: Mean Temperature per Weather Type
# ────────────────────────────────────────────────
plt.figure(figsize=(12, 6))
bar = sns.barplot(
    data=summary_stats,
    x='Description',
    y='mean',
    hue='Description',  # Required to avoid FutureWarning
    palette='coolwarm',
    legend=False
)
bar.set_title("Average Temperature by Weather Description", fontsize=16)

bar.set_ylabel("Average Temperature (°C)")
bar.set_xlabel("Weather Description")
bar.tick_params(axis='x', rotation=45)

# 📝 Add value labels
for i, row in summary_stats.iterrows():
    bar.text(i, row['mean'] + 0.05, f"{row['mean']:.2f}", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────
# 📦 Boxplot: Temperature Distribution by Type
# ────────────────────────────────────────────────
plt.figure(figsize=(14, 7))
sns.boxplot(
    data=combined,
    x='Description',
    y='Temperature',
    hue='Description',  # Required to avoid FutureWarning
    palette='viridis',
    legend=False
)
plt.title("Temperature Distribution by Weather Description", fontsize=16)

plt.xlabel("Weather Description")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
