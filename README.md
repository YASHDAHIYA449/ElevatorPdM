### **ElevatorPdM**



Predictive Maintenance of Elevator Systems using Fault Analysis and Hybrid Neural–Gradient Boosting Residual RUL Modeling



📌 Project Overview



Elevators are safety-critical cyber-physical systems operating under varying mechanical, electrical, and environmental conditions. Unexpected failures can lead to service disruptions, safety hazards, and high maintenance costs.



This project presents a complete, end-to-end Predictive Maintenance (PdM) framework for elevator systems that combines:



Fault detection and severity analysis



Health state characterization



Anomaly detection



Remaining Useful Life (RUL) prediction



Deep learning with residual correction



The final system closely mirrors real industrial PdM pipelines used in high-reliability engineering domains.



🎯 Objectives



Detect fault states (0: No Fault, 1: Fault) with high recall



Analyze fault severity and fault categories



Identify anomalous operating regimes



Characterize health degradation stages



Predict Remaining Useful Life (RUL) accurately



Improve RUL accuracy using Hybrid Neural–Gradient Boosting Residual Learning



Maintain interpretability for safety-critical decision-making



🧠 End-to-End Pipeline Architecture



Raw Time-Series Sensor Data

        ↓

Dataset Quality Checks \& Characterization

        ↓

Exploratory Data Analysis (Fault \& RUL)

        ↓

Pre-processing \& Lifecycle Segmentation

        ↓

Advanced Feature Engineering

        ↓

Anomaly Detection Layer

        ↓

Fault Analysis (Classification)

        ↓

Health Stage Characterization

        ↓

RUL Regression (Baseline → Deep Learning → Residual Learning)



📊 Dataset Description



Source: Public GitHub repository – ElevatorFaultDetection



Geography: High-rise commercial buildings, Tokyo, Japan



Duration: January 2020 – November 2024 (~5 years)



Sampling Frequency: Hourly



System: IoT-enabled elevator monitoring system



Data Characteristics



Multivariate time-series sensor data



Mechanical, electrical, environmental, and maintenance features



Fully supervised labels



Target Variables



Fault\_State → Binary classification (0, 1)



RUL → Continuous regression



🔍 Exploratory Data Analysis (EDA)

🧪 Fault Analysis — Univariate Insights



Most frequent fault categories:



Door Failure



Motor Malfunction



Faults are relatively rare, confirming the need for imbalance-aware modeling



Minor faults dominate, implying that early detection can prevent escalation



Class Imbalance



Fault\_State = 0 overwhelmingly dominates the dataset



Fault\_State = 1 represents a very small fraction



Fault severity analysis confirms:



Minor faults ≫ Moderate ≫ Critical



🔗 Fault Analysis — Bivariate \& Multivariate Insights



Vibration Level vs Fault State



Distribution nearly identical for fault and non-fault



Vibration alone is not sufficient for fault prediction



Motor Current vs Fault State



PDF overlap shows weak discriminative power



Maintenance History



Independent of “days since last service”



Treated as a low-resolution categorical count, not a continuous variable



Ambient Temperature vs Humidity (Critical Fault Density)



Highest density of critical faults occurs at:



Temperature: 50–70



Humidity: 50–100



These variables are strong predictors of fault occurrence



Poor predictors of fault severity level



Component Age vs Motor Voltage vs Fault State



Bimodal voltage regimes (e.g., idle vs active)



Faults occur in both regimes → voltage is not isolating faults



Health Index vs Time



Non-fault state: stable and smooth



Fault state: highly volatile with extreme spikes



Health index volatility is a strong fault indicator



Smoothed Vibration vs Time



Long-term averages are similar



Monthly mean vibration > 1.10 strongly indicates fault state

→ actionable threshold for alarms



⏳ RUL Analysis

📈 Univariate Insights



RUL Distribution



Clearly bimodal:



~1000 RUL → harsh operating conditions / accelerated degradation



~2200 RUL → nominal operating population



Pulley Wear



Also bimodal



Highest density around wear ≈ 65



Vibration Level



Numerous high-value outliers (up to ~3.8)



Strongly associated with low RUL



Component Age



Large population near or beyond design life



Critical for realistic RUL modeling



🔬 Multivariate Insights



RUL vs Vibration vs Ambient Temperature



Temperature is the dominant degradation accelerator



At extreme temperatures (≥120):



Strong linear negative RUL–vibration relationship



Vibration becomes a reliable RUL predictor



RUL vs Speed vs Load



Speed alone is a poor predictor



Cumulative stress matters more



RUL vs Load Accumulation vs High-Speed Hours



Strong negative degradation curve



Confirms total work done is a primary RUL driver



RUL vs Pulley Wear Accumulation vs Lubrication



Weak but consistent negative trend



Dataset dominated by low wear states (healthy population)



⚙️ Pre-processing \& Lifecycle Segmentation

Component Lifecycle Identification



Component\_ID\_Proxy



Derived using differencing on Component\_Age



Negative jump → new lifecycle begins



Enables correct RUL reset handling



Environmental Categorization



Temp\_Category



Low / Moderate / High / Extreme



Used for conditional degradation analysis



Accumulated Stress Features



High\_Speed\_Hours



Pulley\_Wear\_Accumulated



Load\_Accumulation



Cumulative features grouped by Component\_ID\_Proxy



Reset at lifecycle boundaries



Capture total stress exposure, not instantaneous values



🧩 Feature Engineering for RUL Regression



Initial ML models failed to achieve positive R² → feature representation was the bottleneck.



Rolling Window Feature Strategy



Windows Used:

\[3, 6, 12, 24, 48, 120, 240]

(Short-term to long-term horizons)



Generated Features



Rolling Mean → trend smoothing



Rolling Std → instability \& volatility



Rolling Min / Max → extreme behavior



Rolling Last → most recent system state



Slope → direction \& rate of degradation



This transformation enabled the models to learn degradation dynamics, not raw sensor noise.



🚨 Anomaly Detection \& Health Characterization



Dataset quality checks



Early abnormal regime detection



Health stage characterization



SHAP explainability for trust and transparency



🧠 Fault Analysis Models

Models Evaluated



Logistic Regression



Decision Tree



Random Forest



XGBoost



SMOTE for class imbalance



Metrics



Accuracy



Precision



Recall (Fault = 1 prioritized)



F1-score



Tree-based models achieved strong recall, suitable for safety-critical fault detection.



⏳ RUL Regression Modeling



🥉 Baseline Models



Linear Regression



Decision Tree



Random Forest



SVR



CatBoost



LightGBM



XGBoost



Performance:

R² ≈ 0.33–0.35



🤖 Deep Learning Models



🥈 Model - Delta-Based Windowed LSTM RUL Degradation Model



Neural regressors trained in log-RUL space



Captured global degradation patterns



Performance:

R² ≈ 0.53–0.55



🥇 Model — Hybrid Neural–Gradient Boosting Residual RUL Model



Architecture



Neural Network



Learns smooth global degradation trends



Predicts log-RUL



Gradient Boosting Residual Model



Trained on prediction errors



Corrects non-linear residual structure



Final Performance



R² ≈ 0.72



Substantial reduction in RMSE and MAE



Strong generalization across operating regimes



🔎 Explainability



SHAP applied to fault and RUL models



Identification of dominant degradation drivers



Transparent decision support for maintenance planning



📐 Evaluation Metrics

Fault Analysis



Accuracy



Precision



Recall



F1-score



RUL Regression



RMSE



MAE



R²



🛠️ Tech Stack



Language: Python



Data: Pandas, NumPy



ML: Scikit-learn, XGBoost, LightGBM, CatBoost



Deep Learning: TensorFlow, PyTorch



Explainability: SHAP



Visualization: Matplotlib, Seaborn



Environment: Jupyter Notebook, Google Colab



🏁 Key Contributions



✔ Full industrial-grade PdM pipeline

✔ Deep EDA driving feature design

✔ Lifecycle-aware preprocessing

✔ Rolling window degradation modeling

✔ Hybrid residual learning achieving state-of-the-art RUL accuracy

✔ Explainable AI for safety-critical systems

