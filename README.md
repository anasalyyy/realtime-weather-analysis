# 🌤️ Real-Time Weather Pattern Analysis

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)

A real-time machine learning project that analyzes and forecasts weather patterns using sequential temperature and weather description data.

---

## 📊 Dataset Description

📁 **File:** `real_time_weather_dataset_1000x1000.xlsx`  
Each row contains:
- 🔥 `Temp_1` to `Temp_100`: 100 sequential temperature readings.
- 🌤️ `Description_1` to `Description_100`: Corresponding weather descriptions like `clear sky`, `heavy rain`, etc.

---

## 🚀 Problem Statements

### 🔍 P1: Weather Pattern Classification
📌 **Goal:** Predict the overall weather label (e.g., Sunny, Cloudy, Rainy) from the sequence.  
🛠️ **Approach:** ML classification using temperature + description embeddings.

---

### 📉 P2: Temperature Anomaly Detection
📌 **Goal:** Detect sudden spikes/drops indicating faulty sensors or extreme weather.  
🛠️ **Approach:** Z-score, Isolation Forest, Autoencoder-based detection.

---

### ⛅ P3: Forecasting Next Weather Description
📌 **Goal:** Predict `Description_101` from previous 100 steps.  
🛠️ **Approach:** Sequence modeling (LSTM, GRU, Transformer).

---

### 📈 P4: Correlation Between Temp and Descriptions
📌 **Goal:** Analyze how descriptions (e.g., fog, rain) correlate with average/variance in temperatures.  
🛠️ **Approach:** Aggregations, visualizations, correlation heatmaps.

---

### 🌡️ P5: Clustering Weather Profiles
📌 **Goal:** Discover patterns like `hot & dry`, `cool & rainy` using unsupervised learning.  
🛠️ **Approach:** KMeans, PCA/t-SNE visualization.

---

## 🧠 Project Structure

```bash
weather-analysis/
│
├── src/
│   ├── data_loader.py
│   ├── weather_pattern_classification.py      # P1
│   ├── temperature_anomaly_detection.py       # P2
│   ├── forecast_weather_description.py        # P3
│   ├── description_temperature_correlation.py # P4
│   └── weather_profile_clustering.py          # P5
│
├── real_time_weather_dataset_1000x1000.xlsx
├── main.py
├── requirements.txt
├── README.md
└── .gitignore

## ⚙️ Setup & Installation

### 🪟 Step-by-step Windows PowerShell

```powershell
# 1. Clone the repository
git clone https://github.com/anasalyyy/Real-Time-Weather-Analysis.git
cd Real-Time-Weather-Analysis

# 3. Install all required libraries
pip install -r requirements.txt

# 4. Run the main program
python main.py

## 👨‍💻 Author

**Name:** Anas Ali  
**Gmail:** anaskhanop007@gmail.com




