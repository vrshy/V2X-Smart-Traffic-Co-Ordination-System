"""Generate synthetic traffic data for the V2X Smart Traffic project."""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_FILE = DATA_DIR / "traffic_data.csv"
NUM_SAMPLES = 3000
RANDOM_SEED = 42


def get_signal(density: float, emergency: int) -> str:
    """Return a simple illustrative signal state."""
    if emergency:
        return "GREEN"
    if density >= 0.70:
        return "GREEN"
    if density >= 0.40:
        return "YELLOW"
    return "RED"


def get_congestion(density: float) -> str:
    """Convert traffic density into a readable congestion label."""
    if density >= 0.70:
        return "High"
    if density >= 0.40:
        return "Medium"
    return "Low"


def generate_traffic_data(num_samples: int = NUM_SAMPLES) -> pd.DataFrame:
    """Create reproducible synthetic traffic data at 10-minute intervals."""
    rng = np.random.default_rng(RANDOM_SEED)
    times = pd.date_range("2025-01-01 00:00:00", periods=num_samples, freq="10min")

    # Create two daily rush-hour peaks, with a smaller random variation.
    minutes_in_day = (np.arange(num_samples) * 10) % (24 * 60)
    morning_peak = np.exp(-((minutes_in_day - 8 * 60) / 150) ** 2)
    evening_peak = np.exp(-((minutes_in_day - 18 * 60) / 180) ** 2)
    daily_pattern = 0.18 + 0.48 * morning_peak + 0.50 * evening_peak

    density = np.clip(daily_pattern + rng.normal(0, 0.10, num_samples), 0.05, 1.0)
    vehicle_count = np.clip(
        (density * 180 + rng.normal(0, 12, num_samples)).round(), 5, 220
    ).astype(int)
    emergency = (rng.random(num_samples) < 0.04).astype(int)
    average_speed = np.clip(
        70 - density * 48 + rng.normal(0, 4, num_samples), 5, 80
    ).round(1)

    frame = pd.DataFrame(
        {
            "Time": times,
            "Time_Of_Day": times.strftime("%H:%M"),
            "Vehicle_Count": vehicle_count,
            "Traffic_Density": density.round(3),
            "Signal_Status": [
                get_signal(float(d), int(e)) for d, e in zip(density, emergency)
            ],
            "Emergency_Vehicle": emergency,
            "Average_Speed": average_speed,
            "Congestion_Level": [get_congestion(float(d)) for d in density],
        }
    )
    return frame


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic = generate_traffic_data()
    traffic.to_csv(OUTPUT_FILE, index=False)
    print(f"Created {len(traffic):,} synthetic traffic records.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
