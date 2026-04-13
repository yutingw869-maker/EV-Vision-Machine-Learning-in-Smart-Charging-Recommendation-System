# charge.py — Physics-informed features + Random Forest to directly predict charging time, with /predict endpoint

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =================== Flask setup ===================
app = Flask(__name__)
CORS(app)   # allow front-end (e.g. http://127.0.0.1:5500) to access http://127.0.0.1:5000

# =================== 1. Load data ===================
# If your file is not at this path, change to your own path
df = pd.read_excel("ev_charging_new.xlsx")
print("Raw data shape:", df.shape)

target_col = "charging_duration"

# =================== 2. Construct weather_state ===================
def get_weather_state(row):
    if row.get("is_cold_weather", 0) == 1:
        return "cold"
    elif row.get("is_hot_weather", 0) == 1:
        return "hot"
    else:
        return "normal"

df["weather_state"] = df.apply(get_weather_state, axis=1)

# =================== 3. Physics-related features: ΔSOC, E_need, P_nominal, T_base ===================
# ΔSOC
df["delta_soc"] = df["soc_end"] - df["soc_start"]
# Required energy to charge (kWh)
df["energy_to_charge"] = df["battery_capacity"] * df["delta_soc"] / 100.0
df["E_need"] = df["energy_to_charge"]

# If there is no charging_rate column, estimate it as energy_consumed / charging_duration
if ("charging_rate" not in df.columns) or df["charging_rate"].isna().all():
    valid = df[target_col] > 0
    df.loc[valid, "charging_rate"] = (
        df.loc[valid, "energy_consumed"] / df.loc[valid, target_col]
    )

rate_valid = df["charging_rate"] > 0

# 1) Typical battery capacity per vehicle model (used at prediction time)
capacity_by_model = (
    df.groupby("vehicle_model")["battery_capacity"]
    .median()
    .to_dict()
)
global_capacity = df["battery_capacity"].median()

# 2) Typical nominal power P_nominal per (charger_type, is_fast_charging)
group_cols_power = ["charger_type", "is_fast_charging"]
power_table = (
    df[rate_valid]
    .groupby(group_cols_power)["charging_rate"]
    .median()
    .to_dict()
)
global_power = df.loc[rate_valid, "charging_rate"].median()

def get_P_nominal(row):
    key = (row["charger_type"], row["is_fast_charging"])
    return power_table.get(key, global_power)

df["P_nominal"] = df.apply(get_P_nominal, axis=1)

# 3) Physics-based baseline time
mask = (df["E_need"] > 0) & (df["P_nominal"] > 0) & (df[target_col] > 0)
data = df.loc[mask].copy()

data["T_base"] = data["E_need"] / data["P_nominal"]
data = data[(data["T_base"] > 0.01) & (data["T_base"] < 24)].copy()

print("Samples used for time model:", len(data))
print(f"T_base range: [{data['T_base'].min():.3f}, {data['T_base'].max():.3f}] hours")

# =================== 4. Build feature set for directly predicting time ===================
candidate_features = [
    "is_fast_charging",
    "vehicle_model",
    "vehicle_age",
    "charger_type",
    "weather_state",
    "delta_soc",
    "battery_capacity",
    "E_need",
    "P_nominal",
    "T_base",
    "charging_rate",
    "c_rate",
    "temperature",
    "station_id",
    "user_type",
    "day_of_week",
    "time_of_day",
]

feature_cols = [c for c in candidate_features if c in data.columns]

data_time = data[feature_cols + [target_col]].dropna().copy()
X = data_time[feature_cols]
y_time = data_time[target_col]

print("Feature columns used to directly predict time:", feature_cols)
print("Valid sample count:", len(data_time))

# Split numerical / categorical features
numeric_candidate = [
    "is_fast_charging",
    "vehicle_age",
    "delta_soc",
    "battery_capacity",
    "E_need",
    "P_nominal",
    "T_base",
    "charging_rate",
    "c_rate",
    "temperature",
]
numeric_features = [c for c in feature_cols if c in numeric_candidate]
categorical_features = [c for c in feature_cols if c not in numeric_features]

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

rf_regressor = RandomForestRegressor(
    n_estimators=800,
    max_depth=12,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

time_model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", rf_regressor),
    ]
)

# =================== 5. Train / test split, training, evaluation ===================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_time, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

print("\nStart training [physics features + direct time] model...")
time_model.fit(X_train, y_train)
print("Model training finished.")

y_pred = time_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n===== Physics features + direct time model performance =====")
print(f"MAE  : {mae:.3f} hours")
print(f"RMSE : {rmse:.3f} hours")
print(f"R^2  : {r2:.3f}")

abs_errors = np.abs(y_test - y_pred)
within_30min = np.mean(abs_errors <= 0.5)
within_1h = np.mean(abs_errors <= 1.0)
print(f"Proportion of samples with |error| ≤ 0.5h: {within_30min*100:.1f}%")
print(f"Proportion of samples with |error| ≤ 1.0h: {within_1h*100:.1f}%")

# =================== 6. Default template row (used at prediction time) ===================
default_template = {}
for col in feature_cols:
    if col in numeric_features:
        default_template[col] = float(data_time[col].median())
    else:
        default_template[col] = data_time[col].mode().iloc[0]

print("\nDefault template (partial):")
for k in list(default_template.keys())[:6]:
    print("  ", k, "=>", default_template[k])


#save model
joblib.dump(time_model, "charging_time_prediction.pkl")
metadata = {
    "capacity_by_model": capacity_by_model,
    "global_capacity": global_capacity,
    "power_table": power_table,
    "global_power": global_power,
    "default_template": default_template,
    "feature_cols": feature_cols,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
}

joblib.dump(metadata, "charging_time_metadata.pkl")

print("\nModel and metadata saved to:")
print("   - charging_time_prediction.pkl")
print("   - charging_time_metadata.pkl")



# =================== 7. Normalize weather string ===================
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

# =================== 8. Make prediction from front-end payload ===================
def make_scientific_prediction(payload: dict):
    """
    Front-end should send JSON with:
      - is_fast_charging (0/1)
      - vehicle_model
      - vehicle_age
      - charger_type
      - weather        ("cold" / "hot" / "normal")
      - soc_start      (0-100)
      - soc_end        (0-100, > soc_start)
    """
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
        if k not in payload:
            raise ValueError(f"Missing field: {k}")

    is_fast = int(payload["is_fast_charging"])
    vehicle_model = str(payload["vehicle_model"])
    vehicle_age = float(payload["vehicle_age"])
    charger_type = str(payload["charger_type"])
    weather_state = normalize_weather_choice(payload["weather"])
    soc_start = float(payload["soc_start"])
    soc_end = float(payload["soc_end"])

    soc_start = max(0.0, min(100.0, soc_start))
    soc_end = max(0.0, min(100.0, soc_end))
    if soc_end <= soc_start:
        raise ValueError("End SOC must be greater than start SOC.")

    # 1) Battery capacity: use typical value for this vehicle model
    if vehicle_model in capacity_by_model:
        battery_capacity = float(capacity_by_model[vehicle_model])
    else:
        battery_capacity = float(global_capacity)

    # 2) Required energy to charge
    delta_soc = soc_end - soc_start
    E_need = battery_capacity * delta_soc / 100.0

    # 3) Typical nominal power
    key_p = (charger_type, is_fast)
    P_nominal = power_table.get(key_p, global_power)

    if E_need <= 0 or P_nominal <= 0:
        raise ValueError("Cannot compute a valid baseline time from the given input.")

    # 4) Physics-based baseline time
    T_base = E_need / P_nominal

    # 5) Approximate charging_rate / C-rate / temperature at prediction time
    charging_rate_est = P_nominal
    c_rate_est = charging_rate_est / battery_capacity if battery_capacity > 0 else 0.0
    if weather_state == "cold":
        temp_est = 5.0
    elif weather_state == "hot":
        temp_est = 30.0
    else:
        if "temperature" in numeric_features:
            temp_est = float(data_time["temperature"].median())
        else:
            temp_est = 20.0

    # 6) Assemble one feature row for the model
    row = default_template.copy()

    def set_if_used(col, value):
        if col in feature_cols:
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

    input_df = pd.DataFrame([row], columns=feature_cols)

    # 7) Use the trained model to directly predict charging time
    T_pred = float(time_model.predict(input_df)[0])

    # For display: correction factor k = T_pred / T_base
    k_pred = float(T_pred / T_base) if T_base > 0 else 1.0

    return {
        "predicted_hours": T_pred,
        "base_hours": float(T_base),
        "correction_factor": k_pred,
        "energy_to_charge": float(E_need),
        "nominal_power": float(P_nominal),
    }

# =================== 9. Flask routes ===================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data_json = request.get_json()
        if data_json is None:
            return jsonify({"error": "Please send a JSON request body."}), 400

        result = make_scientific_prediction(data_json)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/")
def index():
    return "Physics-informed + Random Forest time prediction model is running. Call /predict from the front-end."

if __name__ == "__main__":
    # Recommended: run in your .conda environment, e.g. `python charge.py`
    app.run(host="127.0.0.1", port=5000, debug=True)
