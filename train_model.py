import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------
# CREATE MODEL FOLDER
# ---------------------------------------------------

os.makedirs("models", exist_ok=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = pd.read_csv("data/traffic_data.csv")

# ---------------------------------------------------
# ENCODE SIGNAL STATUS
# ---------------------------------------------------

signal_map = {
    "RED": 0,
    "YELLOW": 1,
    "GREEN": 2
}

df["Signal_Encoded"] = df["Signal_Status"].map(signal_map)

# ---------------------------------------------------
# TARGET = NEXT time step's density
# ---------------------------------------------------
# FIX: before, the target was the SAME column as a feature
# (Traffic_Density), so the model just copied the answer
# (data leakage). Now each row predicts the NEXT row's density.
# shift(-1) moves the column up by one row.

df["Next_Density"] = df["Traffic_Density"].shift(-1)

# The last row has no "next" value, so drop it
df = df.dropna(subset=["Next_Density"])

# ---------------------------------------------------
# FEATURES (what we know NOW)
# ---------------------------------------------------

FEATURES = [
    "Vehicle_Count",
    "Traffic_Density",
    "Average_Speed",
    "Emergency_Vehicle",
    "Signal_Encoded"
]

X = df[FEATURES]
y = df["Next_Density"]

# ---------------------------------------------------
# SPLIT DATA (chronological)
# ---------------------------------------------------
# For time-ordered data we train on the past and test on the
# future, so no random shuffling.

split_point = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# ---------------------------------------------------
# BASELINE (a "dumb" guess to compare against)
# ---------------------------------------------------

baseline_pred = [y_train.mean()] * len(y_test)
baseline_mae = mean_absolute_error(y_test, baseline_pred)

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------------------
# EVALUATION
# ---------------------------------------------------

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"Baseline MAE (always predict the average): {baseline_mae:.4f}")
print(f"Model MAE:                                 {mae:.4f}")

if mae >= baseline_mae:
    print(
        "Note: the model is not beating the baseline. "
        "The simulated data is random each step, so the next "
        "value is not really predictable."
    )

# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------

joblib.dump(model, "models/traffic_model.pkl")

print("Traffic AI model trained and saved!")
