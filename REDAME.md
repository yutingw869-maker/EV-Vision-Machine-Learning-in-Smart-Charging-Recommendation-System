# environment independence:
Flask
Flask-Cors
joblib
lightgbm
matplotlib
numpy
pandas
plotly
scikit-learn
seaborn

# data processing:
Original data is "ev_charging_patterns.xlsx"
run cleaning.py    you can get "ev_charging_cleaned.xlsx"
run cleanning2.py  you can get "ev_charging_new.xlsx"

# visiualization:
run visiualization.py   you can get some png files about visiualizations

# model:
run charge.py                      you can get "charging_time_prediction.pkl" and "charging_time_metadata.pkl"
run stress_model.py                you can get "stress_model.joblib"
run Charging_power_recommend.py    you can get "charging_power_model.pkl" and "feature_order.pkl"

# test API and open E-vision platform:
unsure you have got all .pkl files and .joblib file
stop the terminal, if you still running the model
run app.py      you can see API like http://127.0.0.1:5000, open it.
