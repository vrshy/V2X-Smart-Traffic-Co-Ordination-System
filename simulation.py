import numpy as np
import pandas as pd
import os
import random

# ---------------------------------------------------
# CREATE DATA FOLDER
# ---------------------------------------------------

os.makedirs("data", exist_ok=True)

# ---------------------------------------------------
# SETTINGS
# ---------------------------------------------------

num_samples = 3000

time = np.arange(num_samples)

# ---------------------------------------------------
# DATA STORAGE
# ---------------------------------------------------

vehicle_count = []

traffic_density = []

signal_status = []

emergency_vehicle = []

average_speed = []

congestion_level = []

# ---------------------------------------------------
# MAIN SIMULATION LOOP
# ---------------------------------------------------

for i in range(num_samples):

    # Vehicle Count
    vehicles = random.randint(5, 100)

    # Traffic Density
    density = min(
        vehicles / 100,
        1.0
    )

    # Emergency Vehicle
    emergency = random.choice(
        [0, 0, 0, 1]
    )

    # Signal Logic
    if emergency == 1:

        signal = "GREEN"

    elif density > 0.7:

        signal = "GREEN"

    elif density > 0.4:

        signal = "YELLOW"

    else:

        signal = "RED"

    # Average Speed
    avg_speed = max(
        15,
        80 - (density * 50)
    )

    # Congestion Level
    if density > 0.7:

        congestion = "HIGH"

    elif density > 0.4:

        congestion = "MEDIUM"

    else:

        congestion = "LOW"

    # Store Data
    vehicle_count.append(vehicles)

    traffic_density.append(
        round(density, 2)
    )

    signal_status.append(signal)

    emergency_vehicle.append(emergency)

    average_speed.append(
        round(avg_speed, 2)
    )

    congestion_level.append(congestion)

# ---------------------------------------------------
# DATAFRAME
# ---------------------------------------------------

df = pd.DataFrame({

    "Time": time,

    "Vehicle_Count":
        vehicle_count,

    "Traffic_Density":
        traffic_density,

    "Signal_Status":
        signal_status,

    "Emergency_Vehicle":
        emergency_vehicle,

    "Average_Speed":
        average_speed,

    "Congestion_Level":
        congestion_level
})

# ---------------------------------------------------
# SAVE CSV
# ---------------------------------------------------

df.to_csv(
    "data/traffic_data.csv",
    index=False
)

print("V2X Traffic Simulation Data Generated!")
print(df.head())