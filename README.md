# ElevatorPdM

Predictive Maintenance of Elevator Systems using Fault Analysis and Hybrid Neural–Gradient Boosting Residual RUL Modeling

## 📌 Project Overview

Elevators are safety-critical cyber-physical systems operating under varying mechanical, electrical, and environmental conditions. Unexpected failures can lead to service disruptions, safety hazards, and high maintenance costs.

This project presents a complete, end-to-end Predictive Maintenance (PdM) framework for elevator systems that combines:

- Fault detection and severity analysis
- Health state characterization
- Anomaly detection
- Remaining Useful Life (RUL) prediction
- Deep learning with residual correction

The final system closely mirrors real industrial PdM pipelines used in high-reliability engineering domains.

## 🎯 Objectives

- Detect fault states (0: No Fault, 1: Fault) with high recall
- Analyze fault severity and fault categories
- Identify anomalous operating regimes
- Characterize health degradation stages
- Predict Remaining Useful Life (RUL) accurately
- Improve RUL accuracy using Hybrid Neural–Gradient Boosting Residual Learning
- Maintain interpretability for safety-critical decision-making

## 🧠 End-to-End Pipeline Architecture

```text
Raw Time-Series Sensor Data
  → Dataset Quality Checks & Characterization
  → Exploratory Data Analysis (Fault & RUL)
  → Pre-processing & Lifecycle Segmentation
  → Advanced Feature Engineering
  → Anomaly Detection Layer
  → Fault Analysis (Classification)
  → Health Stage Characterization
  → RUL Regression (Baseline → Deep Learning → Residual Learning)
```

## 📊 Dataset Description

The elevator time-series data used in this study was downloaded from a publicly available GitHub repository, **ElevatorFaultDetection**. It comes from advanced elevator monitoring and diagnostic systems deployed across high-rise commercial buildings in Tokyo, Japan. The data spans almost 5 years (January 2020 to November 2024) and includes hourly measurements from a network of IoT sensors embedded within the elevator system. The sensors recorded operational, mechanical, electrical, environmental, and maintenance-related characteristics, forming an enriched dataset for real-world elevator operation.

**Data characteristics**

- Multivariate time-series sensor data
- Mechanical, electrical, environmental, and maintenance features
- Fully supervised labels

**Target variables**

- `Fault_State`: binary classification (0, 1)
- `RUL`: continuous regression

## 🔍 Exploratory Data Analysis (EDA)

### 🧪 Fault Analysis: Univariate Insights

- Most frequent fault categories: **Door Failure** and **Motor Malfunction**
- Faults are relatively rare, confirming the need for imbalance-aware modeling
- Minor faults dominate, implying that early detection can prevent escalation

**Class imbalance**

- `Fault_State = 0` overwhelmingly dominates the dataset
- `Fault_State = 1` represents a very small fraction
- Fault severity analysis confirms: Minor faults ≫ Moderate ≫ Critical

### 🔗 Fault Analysis: Bivariate & Multivariate Insights

| Relationship | Finding |
|:--|:--|
| Vibration Level vs Fault State | Distributions are nearly identical for fault and non-fault; vibration alone is not sufficient for fault prediction |
| Motor Current vs Fault State | PDF overlap shows weak discriminative power |
| Maintenance History | Independent of "days since last service"; treated as a low-resolution categorical count, not a continuous variable |
| Ambient Temperature vs Humidity (critical fault density) | Highest density of critical faults at temperature 50–70 and humidity 50–100. Strong predictors of fault occurrence, poor predictors of severity level |
| Component Age vs Motor Voltage vs Fault State | Bimodal voltage regimes (e.g., idle vs active); faults occur in both, so voltage does not isolate faults |
| Health Index vs Time | Non-fault state is stable and smooth; fault state is highly volatile with extreme spikes. Volatility is a strong fault indicator |
| Smoothed Vibration vs Time | Long-term averages are similar, but a monthly mean vibration > 1.10 strongly indicates a fault state, giving an actionable alarm threshold |

## ⏳ RUL Analysis

### 📈 Univariate Insights

- **RUL distribution:** clearly bimodal. ~1000 RUL indicates harsh operating conditions / accelerated degradation; ~2200 RUL indicates the nominal operating population
- **Pulley wear:** also bimodal, with the highest density around wear ≈ 65
- **Vibration level:** numerous high-value outliers (up to ~3.8), strongly associated with low RUL
- **Component age:** a large population is near or beyond design life, which is critical for realistic RUL modeling

### 🔬 Multivariate Insights

- **RUL vs Vibration vs Ambient Temperature:** temperature is the dominant degradation accelerator. At extreme temperatures (≥120) there is a strong linear negative RUL–vibration relationship, so vibration becomes a reliable RUL predictor
- **RUL vs Speed vs Load:** speed alone is a poor predictor; cumulative stress matters more
- **RUL vs Load Accumulation vs High-Speed Hours:** strong negative degradation curve, confirming that total work done is a primary RUL driver
- **RUL vs Pulley Wear Accumulation vs Lubrication:** weak but consistent negative trend; the dataset is dominated by low-wear (healthy) states

## ⚙️ Pre-processing & Lifecycle Segmentation

**Component lifecycle identification**

- `Component_ID_Proxy` is derived by differencing `Component_Age`
- A negative jump marks the start of a new lifecycle, enabling correct RUL reset handling

**Environmental categorization**

- `Temp_Category` (Low / Moderate / High / Extreme) is used for conditional degradation analysis

**Accumulated stress features**

- `High_Speed_Hours`, `Pulley_Wear_Accumulated`, `Load_Accumulation`
- Cumulative features are grouped by `Component_ID_Proxy` and reset at lifecycle boundaries
- They capture total stress exposure, not instantaneous values

## 🧩 Feature Engineering for RUL Regression

Initial ML models failed to achieve positive R², so feature representation was the bottleneck.

**Rolling window strategy**

- Windows used: `[3, 6, 12, 24, 48, 120, 240]` (short-term to long-term horizons)

**Generated features**

| Feature | Captures |
|:--|:--|
| Rolling Mean | Trend smoothing |
| Rolling Std | Instability & volatility |
| Rolling Min / Max | Extreme behavior |
| Rolling Last | Most recent system state |
| Slope | Direction & rate of degradation |

This transformation enabled the models to learn degradation dynamics rather than raw sensor noise.

## 🚨 Anomaly Detection & Health Characterization

- Dataset quality checks
- Early abnormal regime detection
- Health stage characterization
- SHAP explainability for trust and transparency

## 🧠 Fault Analysis Models

- **Models evaluated:** Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Class imbalance:** handled with SMOTE
- **Metrics:** Accuracy, Precision, Recall (Fault = 1 prioritized), F1-score

Tree-based models achieved strong recall, suitable for safety-critical fault detection.

## ⏳ RUL Regression Modeling

| Tier | Model | Performance |
|:--|:--|:--|
| 🥉 Baseline | Linear Regression, Decision Tree, Random Forest, SVR, CatBoost, LightGBM, XGBoost | R² ≈ 0.33–0.35 |
| 🥈 Deep learning | Delta-Based Windowed LSTM RUL Degradation Model (neural regressors trained in log-RUL space, capturing global degradation patterns) | R² ≈ 0.53–0.55 |
| 🥇 Hybrid | Hybrid Neural–Gradient Boosting Residual RUL Model | R² ≈ 0.72 |

### 🥇 Hybrid Neural–Gradient Boosting Residual RUL Model

**Architecture**

- **Neural network:** learns smooth global degradation trends and predicts log-RUL
- **Gradient boosting residual model:** trained on the neural network's prediction errors to correct non-linear residual structure

**Final performance**

- R² ≈ 0.72
- Substantial reduction in RMSE and MAE
- Strong generalization across operating regimes

## 🔎 Explainability

- SHAP applied to fault and RUL models
- Identification of dominant degradation drivers
- Transparent decision support for maintenance planning

## 📐 Evaluation Metrics

- **Fault analysis:** Accuracy, Precision, Recall, F1-score
- **RUL regression:** RMSE, MAE, R²

## 🛠️ Tech Stack

- **Language:** Python
- **Data:** Pandas, NumPy
- **ML:** Scikit-learn, XGBoost, LightGBM, CatBoost
- **Deep Learning:** TensorFlow, PyTorch
- **Explainability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook, Google Colab

## 🏁 Key Contributions

- ✔ Full industrial-grade PdM pipeline
- ✔ Deep EDA driving feature design
- ✔ Lifecycle-aware preprocessing
- ✔ Rolling window degradation modeling
- ✔ Hybrid residual learning achieving R² ≈ 0.72 on RUL prediction
- ✔ Explainable AI for safety-critical systems
