# EV-Vision-Machine-Learning-in-Smart-Charging-Recommendation-System

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-LightGBM%20%7C%20RandomForest-green)
![Framework](https://img.shields.io/badge/Framework-Flask-red)

E-Vision is an intelligent analytical platform designed to optimize Electric Vehicle (EV) charging efficiency and battery longevity. It provides real-time charging time prediction, battery stress assessment, and personalized charging power recommendations.

## 🚀 Key Modules
- **Charging Time Prediction**: Accurate estimation of duration based on current SOC and environment.
- **Battery Health Evaluation**: Stress modeling to predict potential battery degradation.
- **Personalized Recommendation**: Optimized charging power suggestions to balance efficiency and battery life.

---

## 🛠️ 1. Installation & Environment


## Data Pipeline & Visualization
Data Cleaning: Run cleaning.py to handle outliers and missing values in ev_charging_patterns.xlsx.

Feature Engineering: Run cleanning2.py to generate the enhanced dataset ev_charging_new.xlsx.

Visualization: Run visualization.py to generate analytical charts (saved as PNG files).

## Model Training
Module                     Script                                       Output Files
Duration Prediction,  python charge.py                    "charging_time_prediction.pkl, metadata.pkl"
Stress Assessment,    python stress_model.py              stress_model.joblib
Power Recommendation, python Charging_power_recommend.py, "charging_power_model.pkl, feature_order.pkl"

## Deployment & API Usage
Once the models (.pkl and .joblib) are generated, start the Flask backend:
python app.py
Endpoint: http://127.0.0.1:5000
Interface: The platform provides a RESTful API that accepts JSON inputs and returns real-time EV analytics.
