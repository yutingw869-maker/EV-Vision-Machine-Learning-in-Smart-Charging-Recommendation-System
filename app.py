# app.py — Serve static pages + two prediction APIs

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__, template_folder='templates')
CORS(app)

# =================== Helper: Explanation Generator ===================
def build_explanation(level: str,
                      c_rate: float,
                      T: float,
                      soc_start: float,
                      soc_end: float,
                      delta_soc: float) -> str:
    """
    Generate human-readable advice based on charging parameters.
    """
    msg = f"The overall stress level for this charging session is {level}."

    advice_parts = []

    if c_rate is not None and T is not None:
        if c_rate > 1.5 and T < 5:
            advice_parts.append(
                "This session used a high C-rate fast charge in a low-temperature environment. "
                "Next time, consider reducing the charging power or waiting until the battery warms up before fast charging."
            )
        elif c_rate > 1.5:
            advice_parts.append(
                "The C-rate in this session is relatively high. Consider lowering the charging power to reduce cell aging stress."
            )

    if soc_start is not None and soc_start < 10:
        advice_parts.append(
            "Charging started from a low state of charge (SoC). Frequent deep discharges accelerate battery aging; "
            "try to avoid regularly using the battery below 10%."
        )

    if soc_end is not None and soc_end > 90:
        advice_parts.append(
            "This session ended at a high state of charge. Parking for long periods at high SoC increases chemical stress; "
            "for daily use, consider ending charges around 80%."
        )

    if delta_soc is not None and delta_soc > 70:
        advice_parts.append(
            "This session involved a large change in SoC. Frequent large charge/discharge swings increase battery cycle stress."
        )

    if not advice_parts:
        advice_parts.append(
            "The indicators for this charging session are relatively mild. You can continue with your current charging habits."
        )

    return msg + " " + " ".join(advice_parts)


# =================== Load Models ===================
# --- Charging Time Model ---
CHARGE_MODEL_PATH = "charging_time_prediction.pkl"
CHARGE_METADATA_PATH = "charging_time_metadata.pkl"

if not os.path.exists(CHARGE_MODEL_PATH) or not os.path.exists(CHARGE_METADATA_PATH):
    time_model = None
    capacity_by_model = {}
    global_capacity = 60.0
    power_table = {}
    global_power = 50.0
    default_template_charge = {}
    feature_cols_charge = []
else:
    time_model = joblib.load(CHARGE_MODEL_PATH)
    metadata = joblib.load(CHARGE_METADATA_PATH)
    capacity_by_model = metadata["capacity_by_model"]
    global_capacity = metadata["global_capacity"]
    power_table = metadata["power_table"]
    global_power = metadata["global_power"]
    default_template_charge = metadata["default_template"]
    feature_cols_charge = metadata["feature_cols"]

# --- Stress Model ---
STRESS_MODEL_PATH = "stress_model.joblib"
if not os.path.exists(STRESS_MODEL_PATH):
    raise FileNotFoundError(f"Stress model not found: {STRESS_MODEL_PATH}")
stress_data = joblib.load(STRESS_MODEL_PATH)
stress_model = stress_data['model']
numeric_features_stress = stress_data['numeric_features']
categorical_features_stress = stress_data['categorical_features']
default_template_stress = stress_data['default_template']
COL = stress_data.get('COL', {})

# --- Anomaly Detection Support: Power Recommendation Model ---
POWER_MODEL_PATH = "charging_power_model.pkl"
FEATURE_ORDER_PATH = "feature_order.pkl"

if not os.path.exists(POWER_MODEL_PATH) or not os.path.exists(FEATURE_ORDER_PATH):
    power_recommend_model = None
    feature_order = None
    print("Power recommendation model NOT loaded (optional for anomaly detection)")
else:
    power_recommend_model = joblib.load(POWER_MODEL_PATH)
    feature_order = joblib.load(FEATURE_ORDER_PATH)
    print("Power recommendation model loaded for anomaly context")


# =================== Helper Functions ===================
def normalize_weather_choice(choice: str) -> str:
    if choice is None:
        return "normal"
    c = str(choice).strip().lower()
    if c in ["cold", "cold_weather"]:
        return "cold"
    elif c in ["hot", "hot_weather"]:
        return "hot"
    else:
        return "normal"


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


# =================== Power Recommendation Logic ===================
def recommend_charging_power(model, input_record: dict, feature_order: list):
    try:
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
            "current_power_kW": float(current_power) if current_power is not None else None,
            "power_change_kW": float(round(optimal_power - current_power, 3)) if current_power is not None else None,
            "strategy": strategy,
            "charger_suggestion": charger_suggestion,
        }
    except Exception as e:
        print(f"Error in recommend_charging_power: {e}")
        return {
            "recommended_power_kW": None,
            "current_power_kW": None,
            "power_change_kW": None,
            "strategy": "Recommendation unavailable.",
            "charger_suggestion": "Recommendation unavailable."
        }


# =================== Routes: Web Pages ===================
@app.route("/")
def home():
    return send_from_directory(CURRENT_DIR, "index.html")

@app.route("/about")
def about():
    return send_from_directory(CURRENT_DIR, "about.html")

@app.route("/charging_time_prediction")
def charging_page():
    return send_from_directory(CURRENT_DIR, "charging_time_prediction.html")

@app.route("/battery_stress")
def stress_page():
    return send_from_directory(CURRENT_DIR, "battery_stress.html")

@app.route("/power_detection")
def anomaly_page():
    return send_from_directory(CURRENT_DIR, "power_detection.html")


# =================== API: Charging Time Prediction ===================
@app.route("/predict", methods=["POST"])
def predict_charging_time():
    try:
        if time_model is None:
            return jsonify({"error": "Charging model not loaded."}), 500
        data = request.get_json()
        required_keys = [
            "is_fast_charging",
            "vehicle_model",
            "vehicle_age",
            "charger_type",
            "weather",
            "soc_start",
            "soc_end",
        ]
        for k in required_keys:
            if k not in data:
                raise ValueError(f"Missing field: {k}")

        is_fast = int(data["is_fast_charging"])
        vehicle_model = str(data["vehicle_model"])
        vehicle_age = float(data["vehicle_age"])
        charger_type = str(data["charger_type"])
        weather_state = normalize_weather_choice(data["weather"])
        soc_start = float(data["soc_start"])
        soc_end = float(data["soc_end"])

        soc_start = max(0.0, min(100.0, soc_start))
        soc_end = max(0.0, min(100.0, soc_end))
        if soc_end <= soc_start:
            raise ValueError("End SOC must be greater than start SOC.")

        battery_capacity = float(capacity_by_model.get(vehicle_model, global_capacity))
        delta_soc = soc_end - soc_start
        E_need = battery_capacity * delta_soc / 100.0

        key_p = (charger_type, is_fast)
        P_nominal = power_table.get(key_p, global_power)

        if E_need <= 0 or P_nominal <= 0:
            raise ValueError("Invalid energy or power.")

        T_base = E_need / P_nominal
        charging_rate_est = P_nominal
        c_rate_est = charging_rate_est / battery_capacity if battery_capacity > 0 else 0.0

        if weather_state == "cold":
            temp_est = 5.0
        elif weather_state == "hot":
            temp_est = 30.0
        else:
            temp_est = default_template_charge.get("temperature", 20.0)

        row = default_template_charge.copy()

        def set_if_used(col, value):
            if col in feature_cols_charge:
                row[col] = value

        set_if_used("is_fast_charging", is_fast)
        set_if_used("vehicle_model", vehicle_model)
        set_if_used("vehicle_age", vehicle_age)
        set_if_used("charger_type", charger_type)
        set_if_used("weather_state", weather_state)
        set_if_used("delta_soc", delta_soc)
        set_if_used("battery_capacity", battery_capacity)
        set_if_used("E_need", E_need)
        set_if_used("P_nominal", P_nominal)
        set_if_used("T_base", T_base)
        set_if_used("charging_rate", charging_rate_est)
        set_if_used("c_rate", c_rate_est)
        set_if_used("temperature", temp_est)

        input_df = pd.DataFrame([row], columns=feature_cols_charge)
        T_pred = float(time_model.predict(input_df)[0])
        k_pred = float(T_pred / T_base) if T_base > 0 else 1.0

        return jsonify({
            "predicted_hours": round(T_pred, 3),
            "base_hours": round(float(T_base), 3),
            "correction_factor": round(k_pred, 3),
            "energy_to_charge": round(float(E_need), 3),
            "nominal_power": round(float(P_nominal), 3),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# =================== API: Battery Stress ===================
@app.route("/predict_stress", methods=["POST"])
def predict_stress():
    try:
        payload = request.get_json()
        if payload is None:
            return jsonify({"error": "JSON body required"}), 400

        row = default_template_stress.copy()

        def get_float(key, default=None):
            v = payload.get(key)
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        soc_start = get_float("soc_start")
        soc_end = get_float("soc_end")
        T = get_float("temperature")
        charging_rate = get_float("charging_rate")
        battery_capacity = get_float("battery_capacity")
        energy_consumed = get_float("energy_consumed")
        distance_driven = get_float("distance_driven")
        vehicle_age = get_float("vehicle_age")

        vehicle_model = payload.get("vehicle_model")
        charger_type = payload.get("charger_type")
        user_type = payload.get("user_type")
        station_location = payload.get("station_location")

        # Compute derived features
        delta_soc = soc_end - soc_start if soc_start is not None and soc_end is not None else None
        c_rate = charging_rate / battery_capacity if charging_rate and battery_capacity else None

        is_cold_weather = float(T < 5) if T is not None else 0.0
        is_hot_weather = float(T > 35) if T is not None else 0.0

        ct_low = str(charger_type).lower() if charger_type else ""
        is_fast_charging = float(("fast" in ct_low) or ("dc" in ct_low))

        is_full_charge = float(soc_end > 90) if soc_end is not None else 0.0
        low_soc_flag = float(soc_start < 10) if soc_start is not None else 0.0

        # Fill row
        def set_if_feature(name, value):
            if name in row and value is not None:
                row[name] = value

        set_if_feature(COL.get("soc_start", "soc_start"), soc_start)
        set_if_feature(COL.get("soc_end", "soc_end"), soc_end)
        set_if_feature(COL.get("temperature", "temperature"), T)
        set_if_feature(COL.get("chg_power", "charging_rate"), charging_rate)
        set_if_feature(COL.get("batt_capacity", "battery_capacity"), battery_capacity)
        set_if_feature(COL.get("energy", "energy_consumed"), energy_consumed)
        set_if_feature(COL.get("distance", "distance_driven"), distance_driven)
        set_if_feature(COL.get("vehicle_age", "vehicle_age"), vehicle_age)

        set_if_feature("delta_soc", delta_soc)
        set_if_feature("c_rate", c_rate)
        set_if_feature("is_cold_weather", is_cold_weather)
        set_if_feature("is_hot_weather", is_hot_weather)
        set_if_feature("is_fast_charging", is_fast_charging)
        set_if_feature("is_full_charge", is_full_charge)
        set_if_feature("low_soc_flag", low_soc_flag)

        if vehicle_model:
            set_if_feature(COL.get("vehicle_model", "vehicle_model"), vehicle_model)
        if charger_type:
            set_if_feature(COL.get("charger_type", "charger_type"), charger_type)
        if user_type:
            set_if_feature(COL.get("user_type", "user_type"), user_type)
        if station_location:
            set_if_feature(COL.get("station_location", "station_location"), station_location)

        feature_cols = numeric_features_stress + categorical_features_stress
        X_new = pd.DataFrame([row], columns=feature_cols)

        proba = stress_model.predict_proba(X_new)[0]
        classes = stress_model.named_steps["model"].classes_
        pred_label = str(classes[np.argmax(proba)])
        probs_dict = {str(cls): float(p) for cls, p in zip(classes, proba)}

        # ✅ FIXED: Now uses dynamic explanation!
        explanation = build_explanation(
            level=pred_label,
            c_rate=c_rate,
            T=T,
            soc_start=soc_start,
            soc_end=soc_end,
            delta_soc=delta_soc
        )

        return jsonify({
            "stress_level": pred_label,
            "probs": probs_dict,
            "explanation": explanation,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# =================== API: power recommendation Detection ===================
@app.route("/predict_anomaly", methods=["POST"])
def predict_anomaly():
    data = request.get_json()
    try:
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        required = ["soc_start", "soc_end", "temperature", "charging_rate"]
        for k in required:
            if k not in data:
                return jsonify({"error": f"Missing required field: {k}"}), 400

        soc_start = float(data["soc_start"])
        soc_end = float(data["soc_end"])
        temp = float(data["temperature"])
        charging_rate = float(data["charging_rate"])

        if not (0 <= soc_start <= 100) or not (0 <= soc_end <= 100):
            return jsonify({"error": "SOC must be between 0 and 100"}), 400
        if soc_end <= soc_start:
            return jsonify({"error": "End SOC must be > Start SOC"}), 400

        anomaly_score = 0.0
        reasons = []

        if temp < -10 or temp > 45:
            anomaly_score += 0.4
            reasons.append("Extreme ambient temperature")

        if charging_rate > 2.0 and soc_start > 80:
            anomaly_score += 0.35
            reasons.append("High charging power above 80% SOC")

        delta_soc = soc_end - soc_start
        if delta_soc > 80 and charging_rate > 2.5:
            anomaly_score += 0.3
            reasons.append("Very rapid deep charging")

        recommendation = {}
        optimal_power = None
        if power_recommend_model is not None and feature_order is not None:
            try:
                input_record = {col: float(data.get(col, 0.0)) for col in feature_order}
                recommendation = recommend_charging_power(
                    power_recommend_model, input_record, feature_order
                )
                optimal_power = recommendation.get("recommended_power_kW")
                if optimal_power and optimal_power > 0:
                    deviation = abs(charging_rate - optimal_power) / optimal_power
                    if deviation > 0.5:
                        anomaly_score += 0.25
                        reasons.append("Charging power deviates significantly from optimal")
            except Exception as e:
                print("Recommendation failed:", e)

        threshold = 0.5
        is_anomaly = anomaly_score >= threshold

        response_data = {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 3),
            "confidence": round(anomaly_score, 3),
            "reasons": reasons,
            "message": "Anomaly detected" if is_anomaly else "Normal charging pattern",
            "recommended_power_kW": recommendation.get("recommended_power_kW"),
            "current_power_kW": recommendation.get("current_power_kW"),
            "power_change_kW": recommendation.get("power_change_kW"),
            "strategy": recommendation.get("strategy"),
            "charger_suggestion": recommendation.get("charger_suggestion"),
        }

        return jsonify(response_data)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400


# =================== Run ===================
if __name__ == "__main__":
    print("Starting EV Analytics App")
    print("Pages:")
    print("  http://127.0.0.1:5000/                   → Home")
    print("  http://127.0.0.1:5000/charging_time_prediction → Charging Time")
    print("  http://127.0.0.1:5000/battery_stress      → Battery Stress")
    print("  http://127.0.0.1:5000/power_detection   → Anomaly Detection")
    app.run(host="127.0.0.1", port=5000, debug=False)