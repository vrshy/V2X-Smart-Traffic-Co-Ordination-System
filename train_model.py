"""Train and evaluate a next-step traffic-density prediction model."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "traffic_data.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_FILE = MODEL_DIR / "traffic_model.pkl"

FEATURES = [
    "Time",
    "Vehicle_Count",
    "Traffic_Density",
    "Average_Speed",
    "Emergency_Vehicle",
    "Signal_Encoded",
]


def prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Add model-ready columns and the next record's density as target."""
    data = frame.copy()
    data["Time"] = pd.to_datetime(data["Time"]).astype("int64") // 10**9
    data["Signal_Encoded"] = data["Signal_Status"].map(
        {"RED": 0, "YELLOW": 1, "GREEN": 2}
    ).fillna(0)
    data["Next_Density"] = data["Traffic_Density"].shift(-1)
    return data.dropna(subset=["Next_Density"]).reset_index(drop=True)


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Traffic data not found: {DATA_FILE}\n"
            "Run 'python simulation.py' first."
        )

    raw = pd.read_csv(DATA_FILE)
    data = prepare_data(raw)

    if len(data) < 20:
        raise ValueError("At least 20 traffic records are required to train the model.")

    split_index = int(len(data) * 0.80)
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    x_train, y_train = train[FEATURES], train["Next_Density"]
    x_test, y_test = test[FEATURES], test["Next_Density"]

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    mean_baseline = np.full(len(y_test), y_train.mean())
    carry_forward_baseline = test["Traffic_Density"].to_numpy()

    print("\nModel evaluation (chronological 80/20 split)")
    print("-" * 52)
    print(f"Training rows: {len(train):,}")
    print(f"Testing rows:  {len(test):,}")
    print(f"Mean baseline MAE:          {mean_absolute_error(y_test, mean_baseline):.4f}")
    print(
        "Carry-forward baseline MAE: "
        f"{mean_absolute_error(y_test, carry_forward_baseline):.4f}"
    )
    print(f"Random Forest MAE:           {mean_absolute_error(y_test, predictions):.4f}")
    print(
        "Random Forest RMSE:          "
        f"{np.sqrt(mean_squared_error(y_test, predictions)):.4f}"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved model to: {MODEL_FILE}")
    print("Note: compare the model against both baselines; lower error is better.")


if __name__ == "__main__":
    main()
