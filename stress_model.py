from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===================== Flask basics =====================
app = Flask(__name__)
CORS(app)   # Allow frontend (e.g., http://127.0.0.1:5500 / file://) to access the API

# ===================== 1. Load raw data =====================
FILE_PATH = r"ev_charging_new.xlsx"  # TODO: change to your actual path

print(f"Loading data from: {FILE_PATH}")
df_raw = pd.read_excel(FILE_PATH)
print("Raw shape:", df_raw.shape)

# Column mapping (corresponding to your MATLAB col.xxx)
COL = {
    "soc_start":        "soc_start",
    "soc_end":          "soc_end",
    "temperature":      "temperature",
    "chg_power":        "charging_rate",
    "batt_capacity":    "battery_capacity",
    "energy":           "energy_consumed",
    "distance":         "distance_driven",
    "start_time":       "start_time",
    "end_time":         "end_time",
    "vehicle_model":    "vehicle_model",
    "user_type":        "user_type",
    "charger_type":     "charger_type",
    "station_location": "station_location",
    "vehicle_age":      "vehicle_age",
}

# ===================== 2. Data cleaning (for training) =====================
def preprocess_training_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse time columns
    for col in [COL["start_time"], COL["end_time"]]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            raise ValueError(f"Missing column: {col}")

    # Charging duration (minutes)
    df["ChargingDuration_min"] = (df[COL["end_time"]] - df[COL["start_time"]]).dt.total_seconds() / 60.0

    # Basic anomalies: negative energy, non-positive duration, SOC logic errors
    bad = pd.Series(False, index=df.index)

    if COL["energy"] in df.columns:
        bad |= df[COL["energy"]] < 0

    bad |= df["ChargingDuration_min"] <= 0

    if COL["soc_start"] in df.columns and COL["soc_end"] in df.columns:
        bad |= df[COL["soc_end"]] < df[COL["soc_start"]]

    print("Before cleaning:", len(df))
    df = df.loc[~bad].copy()
    print("After removing abnormal rows:", len(df))

    # Fill numeric missing values with the median
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].isna().any():
                median_val = df[c].median()
                df[c] = df[c].fillna(median_val)

    return df

# ===================== 3. Physical feature engineering (for training & inference) =====================
def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    soc_start = df[COL["soc_start"]]
    soc_end   = df[COL["soc_end"]]
    T         = df[COL["temperature"]]
    P_chg     = df[COL["chg_power"]]
    C_batt    = df[COL["batt_capacity"]]

    # C-rate
    df["c_rate"] = P_chg / C_batt.replace(0, np.nan)
    df["c_rate"] = df["c_rate"].fillna(0)

    # delta SOC
    df["delta_soc"] = soc_end - soc_start

    # Temperature flags
    df["is_cold_weather"] = (T < 5).astype(float)
    df["is_hot_weather"]  = (T > 35).astype(float)

    # Fast-charging flag (if charger_type contains "fast" or "dc")
    if COL["charger_type"] in df.columns:
        chg_type_str = df[COL["charger_type"]].astype(str).str.lower()
        df["is_fast_charging"] = (
            chg_type_str.str.contains("fast") | chg_type_str.str.contains("dc")
        ).astype(float)
    else:
        df["is_fast_charging"] = 0.0

    # High SoC parking & low SoC start flags
    df["is_full_charge"] = (soc_end > 90).astype(float)
    df["low_soc_flag"] = (soc_start < 10).astype(float)

    # Energy intensity (kWh/km)
    if COL["energy"] in df.columns and COL["distance"] in df.columns:
        E = df[COL["energy"]]
        D = df[COL["distance"]].replace(0, np.nan)
        df["energy_per_km"] = (E / D).replace([np.inf, -np.inf], np.nan)
        df["energy_per_km"] = df["energy_per_km"].fillna(df["energy_per_km"].median())
    else:
        df["energy_per_km"] = 0.0

    # Time features: time_of_day + day_of_week
    tStart = df[COL["start_time"]]

    # Time-of-day bucket
    hour_val = tStart.dt.hour
    time_of_day = []
    for h in hour_val:
        if pd.isna(h):
            time_of_day.append("Unknown")
        elif 6 <= h < 12:
            time_of_day.append("Morning")
        elif 12 <= h < 18:
            time_of_day.append("Afternoon")
        elif 18 <= h < 24:
            time_of_day.append("Evening")
        else:
            time_of_day.append("Night")
    df["time_of_day"] = pd.Categorical(time_of_day)

    # Day of week
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_str = []
    for ts in tStart:
        if pd.isna(ts):
            day_str.append("Unknown")
        else:
            day_str.append(weekday_names[ts.weekday()])
    df["day_of_week"] = pd.Categorical(day_str)

    return df

# ===================== 4. Compute stress_score + stress_level =====================
def compute_stress_score(df: pd.DataFrame):
    c_rate    = df["c_rate"]
    T         = df[COL["temperature"]]
    soc_start = df[COL["soc_start"]]
    soc_end   = df[COL["soc_end"]]
    delta_soc = df["delta_soc"]

    N = len(df)

    # 1) C-rate stress
    S_crate = np.maximum(0, c_rate - 1.0)

    # 2) Temperature stress
    S_temp = np.zeros(N)
    idx_cold = T < 10
    idx_hot  = T > 30
    S_temp[idx_cold] = (10 - T[idx_cold]) / 10.0
    S_temp[idx_hot]  = (T[idx_hot] - 30) / 10.0

    # 3) SOC stress
    S_soc = np.zeros(N)
    S_soc[soc_start < 10] += 1.0
    S_soc[soc_end   > 90] += 1.0
    S_soc[delta_soc > 70] += 0.5

    # Overall score
    stress_score = 1.5 * S_crate + 1.0 * S_temp + 1.0 * S_soc

    # Stress level
    stress_level = np.full(N, "Green", dtype=object)
    stress_level[(stress_score >= 2) & (stress_score <= 5)] = "Yellow"
    stress_level[stress_score > 5] = "Red"

    return stress_score, pd.Categorical(stress_level, categories=["Green", "Yellow", "Red"])

# ===================== 5. Build supervised dataset =====================
def build_supervised_dataset(df_feat: pd.DataFrame):
    # Labels
    y = df_feat["stress_level"]

    numeric_features = [
        COL["batt_capacity"],
        COL["energy"],
        COL["soc_start"],
        COL["soc_end"],
        COL["distance"],
        COL["temperature"],
        COL["vehicle_age"],
        "ChargingDuration_min",
        "c_rate",
        "delta_soc",
        "energy_per_km",
        "is_cold_weather",
        "is_hot_weather",
        "is_fast_charging",
        "is_full_charge",
        "low_soc_flag",
        "stress_score",
    ]

    categorical_features = [
        COL["vehicle_model"],
        COL["station_location"],
        COL["charger_type"],
        COL["user_type"],
        "time_of_day",
        "day_of_week",
    ]

    # Keep only columns that actually exist
    all_cols = df_feat.columns.tolist()
    numeric_features = [c for c in numeric_features if c in all_cols]
    categorical_features = [c for c in categorical_features if c in all_cols]

    X = df_feat[numeric_features + categorical_features].copy()

    # Simple missing-value imputation
    for c in numeric_features:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    for c in categorical_features:
        X[c] = X[c].astype(str).fillna("Unknown")

    return X, y, numeric_features, categorical_features

# ===================== 6. Train model (runs on import) =====================
df_clean = preprocess_training_df(df_raw)
df_feat  = add_physical_features(df_clean)
stress_score, stress_level = compute_stress_score(df_feat)
df_feat["stress_score"] = stress_score
df_feat["stress_level"] = stress_level

X, y, numeric_features, categorical_features = build_supervised_dataset(df_feat)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape, "Test size:", X_test.shape)

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", rf_clf),
    ]
)

print("\nTraining RandomForest stress model...")
model.fit(X_train, y_train)
print("Model trained.")

# Simple evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n==== Stress model performance on test set ====")
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# Build default template for prediction (used to fill missing features)
default_template = {}
for col_name in numeric_features:
    default_template[col_name] = float(df_feat[col_name].median())
for col_name in categorical_features:
    default_template[col_name] = str(df_feat[col_name].mode().iloc[0])

print("\nDefault template (part):")
for k in list(default_template.keys())[:6]:
    print("  ", k, "=>", default_template[k])


# Save the trained model to disk
MODEL_SAVE_PATH = "stress_model.joblib"
joblib.dump({
    'model': model,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'default_template': default_template,
    'COL': COL
}, MODEL_SAVE_PATH)
print(f"\nModel saved to: {MODEL_SAVE_PATH}")

# ===================== 7. Explanation text for a single record =====================
def build_explanation(level: str,
                      c_rate: float,
                      T: float,
                      soc_start: float,
                      soc_end: float,
                      delta_soc: float) -> str:
    """
    Explanation logic similar to the MATLAB predict_stress function.
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

# ===================== 8. Convert frontend JSON into a feature row =====================
def build_feature_row_from_payload(payload: dict) -> (pd.DataFrame, dict):
    """
    Use one record from the frontend to build a single-row feature DataFrame (same columns as training).
    Also return a dict with c_rate / temperature / SoC, etc., for explanation.
    """

    # Start from default template (so unspecified features are filled with typical values)
    row = default_template.copy()

    # ---- Parse basic inputs ----
    def get_float(key, default=None):
        v = payload.get(key, None)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    soc_start = get_float("soc_start", None)
    soc_end   = get_float("soc_end", None)
    T         = get_float("temperature", None)
    charging_rate = get_float("charging_rate", None)
    battery_capacity = get_float("battery_capacity", None)
    energy_consumed  = get_float("energy_consumed", None)
    distance_driven  = get_float("distance_driven", None)
    vehicle_age      = get_float("vehicle_age", None)

    vehicle_model    = payload.get("vehicle_model", None)
    charger_type     = payload.get("charger_type", None)
    user_type        = payload.get("user_type", None)
    station_location = payload.get("station_location", None)

    start_time_str = payload.get("start_time", "")
    end_time_str   = payload.get("end_time", "")

    # ---- Time handling & duration ----
    start_dt = pd.to_datetime(start_time_str, errors="coerce") if start_time_str else pd.NaT
    end_dt   = pd.to_datetime(end_time_str, errors="coerce") if end_time_str else pd.NaT

    if not pd.isna(start_dt) and not pd.isna(end_dt):
        duration_min = (end_dt - start_dt).total_seconds() / 60.0
    else:
        # If no start/end time provided, estimate duration using E/P
        if energy_consumed is not None and charging_rate and charging_rate > 0:
            duration_min = energy_consumed / charging_rate * 60.0
        else:
            duration_min = default_template.get("ChargingDuration_min", 60.0)

    # ---- Derived physical features ----
    if soc_start is not None and soc_end is not None:
        delta_soc = soc_end - soc_start
    else:
        delta_soc = None

    if charging_rate is not None and battery_capacity:
        c_rate = charging_rate / battery_capacity
    else:
        c_rate = None

    if energy_consumed is not None and distance_driven and distance_driven > 0:
        energy_per_km = energy_consumed / distance_driven
    else:
        energy_per_km = default_template.get("energy_per_km", 0.0)

    # Temperature flags
    is_cold_weather = float(T < 5) if T is not None else 0.0
    is_hot_weather  = float(T > 35) if T is not None else 0.0

    # Fast-charging flag
    if charger_type is not None:
        ct_low = str(charger_type).lower()
        is_fast_charging = float(("fast" in ct_low) or ("dc" in ct_low))
    else:
        is_fast_charging = 0.0

    is_full_charge = float(soc_end > 90) if soc_end is not None else 0.0
    low_soc_flag   = float(soc_start < 10) if soc_start is not None else 0.0

    # time_of_day and day_of_week
    if not pd.isna(start_dt):
        h = start_dt.hour
        if 6 <= h < 12:
            time_of_day = "Morning"
        elif 12 <= h < 18:
            time_of_day = "Afternoon"
        elif 18 <= h < 24:
            time_of_day = "Evening"
        else:
            time_of_day = "Night"
        weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_of_week = weekday_names[start_dt.weekday()]
    else:
        time_of_day = default_template.get("time_of_day", "Unknown")
        day_of_week = default_template.get("day_of_week", "Unknown")

    # ---- Write these into the row (only if feature exists) ----
    def set_if_feature(name, value):
        if name in row and value is not None:
            row[name] = value

    set_if_feature(COL["soc_start"], soc_start)
    set_if_feature(COL["soc_end"], soc_end)
    set_if_feature(COL["temperature"], T)
    set_if_feature(COL["chg_power"], charging_rate)
    set_if_feature(COL["batt_capacity"], battery_capacity)
    set_if_feature(COL["energy"], energy_consumed)
    set_if_feature(COL["distance"], distance_driven)
    set_if_feature(COL["vehicle_age"], vehicle_age)

    set_if_feature("ChargingDuration_min", duration_min)
    set_if_feature("delta_soc", delta_soc)
    set_if_feature("c_rate", c_rate)
    set_if_feature("energy_per_km", energy_per_km)
    set_if_feature("is_cold_weather", is_cold_weather)
    set_if_feature("is_hot_weather", is_hot_weather)
    set_if_feature("is_fast_charging", is_fast_charging)
    set_if_feature("is_full_charge", is_full_charge)
    set_if_feature("low_soc_flag", low_soc_flag)

    if vehicle_model is not None:
        set_if_feature(COL["vehicle_model"], vehicle_model)
    if charger_type is not None:
        set_if_feature(COL["charger_type"], charger_type)
    if user_type is not None:
        set_if_feature(COL["user_type"], user_type)
    if station_location is not None:
        set_if_feature(COL["station_location"], station_location)

    set_if_feature("time_of_day", time_of_day)
    set_if_feature("day_of_week", day_of_week)

    # Build DataFrame (column order consistent with training)
    feature_cols = numeric_features + categorical_features
    X_new = pd.DataFrame([row], columns=feature_cols)

    extra_info = {
        "c_rate": c_rate,
        "temperature": T,
        "soc_start": soc_start,
        "soc_end": soc_end,
        "delta_soc": delta_soc,
    }
    return X_new, extra_info

# ===================== 9. Flask routes =====================
@app.route("/predict_stress", methods=["POST"])
def predict_stress_api():
    try:
        payload = request.get_json()
        if payload is None:
            return jsonify({"error": "Please send JSON body."}), 400

        X_new, info = build_feature_row_from_payload(payload)

        # Model prediction
        proba = model.predict_proba(X_new)[0]
        classes = model.named_steps["model"].classes_
        idx_max = int(np.argmax(proba))
        pred_label = str(classes[idx_max])

        # Probability dictionary
        probs_dict = {}
        for cls, p in zip(classes, proba):
            probs_dict[str(cls)] = float(p)

        explanation = build_explanation(
            pred_label,
            info["c_rate"],
            info["temperature"],
            info["soc_start"],
            info["soc_end"],
            info["delta_soc"],
        )

        return jsonify({
            "stress_level": pred_label,
            "probs": probs_dict,
            "explanation": explanation,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/")
def index():
    return "EV stress model is running. Use POST /predict_stress."

# ===================== main =====================
if __name__ == "__main__":
    # Run in your environment with: python ev_stress_api.py
    app.run(host="127.0.0.1", port=5000, debug=True)
