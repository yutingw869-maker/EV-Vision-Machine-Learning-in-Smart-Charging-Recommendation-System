from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

app = Flask(__name__)

df = pd.read_excel("ev_charging_new.xlsx")

feature_cols = [
    "soc_start", "soc_end", "temperature", "dod",
    "charging_rate", "energy_consumed", "vehicle_age",
    "is_fast_charging", "is_cold_weather", "is_hot_weather",
    "distance_driven", "energy_per_km"
]
target_col = "charging_rate"

X = df[feature_cols]
y = df[target_col]
feature_order = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.04,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9
)
model.fit(X_train, y_train)

#save model
joblib.dump(model, "charging_power_model.pkl")
joblib.dump(feature_order, "feature_order.pkl")
print("Model and feature order saved.")

# Recommendation function
def recommend_charging_power(model, input_record: dict, feature_order: list):
    X_input = pd.DataFrame([input_record])[feature_order]
    optimal_power = model.predict(X_input)[0]
    current_power = input_record.get("charging_rate", None)

    if current_power is not None:
        if optimal_power < current_power:
            strategy = "Recommend reducing charging power to reduce battery stress."
        elif optimal_power > current_power:
            strategy = "You can increase charging power to improve efficiency."
        else:
            strategy = "Current power is already near optimal."
    else:
        strategy = "Current charging power not provided."

    if optimal_power >= 30:
        charger_suggestion = "Recommend DC fast charging."
    elif optimal_power < 10:
        charger_suggestion = "Recommend Level 2 slow charging."
    else:
        charger_suggestion = "Current charging method is appropriate."

    return {
        "recommended_power_kW": float(round(optimal_power, 3)),
        "current_power_kW": float(current_power) if current_power else None,
        "power_change_kW": float(round(optimal_power - current_power, 3)) if current_power else None,
        "strategy": strategy,
        "charger_suggestion": charger_suggestion,
    }

# 3. Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()
        # Convert all numeric fields
        for key in input_data:
            input_data[key] = float(input_data[key])

        result = recommend_charging_power(model, input_data, feature_order)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

# Run app
if __name__ == '__main__':
    app.run(debug=True)
