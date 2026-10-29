"""Streamlit dashboard for the V2X Smart Traffic Coordination System.

All data and vehicle movements in this demo are simulated. This is not a live
traffic-control system and must not be used to control real traffic signals.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "traffic_data.csv"
MODEL_FILE = PROJECT_DIR / "models" / "traffic_model.pkl"

FEATURES = [
    "Time",
    "Vehicle_Count",
    "Traffic_Density",
    "Average_Speed",
    "Emergency_Vehicle",
    "Signal_Encoded",
]

st.set_page_config(
    page_title="V2X Smart Traffic Coordination",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 V2X Smart Traffic Coordination System")
st.caption(
    "Portfolio simulation using synthetic traffic data. "
    "V2X communication, vehicle locations, and routes shown here are illustrative."
)


@st.cache_data
def load_data(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time  # Included in the cache key so updated CSVs reload.
    return pd.read_csv(path, parse_dates=["Time"])


@st.cache_resource
def load_model(path: str, modified_time: float):
    del modified_time
    return joblib.load(path)


def signal_for(density: float, emergency: bool) -> str:
    if emergency:
        return "GREEN"
    if density >= 0.70:
        return "GREEN"
    if density >= 0.40:
        return "YELLOW"
    return "RED"


def congestion_for(density: float) -> str:
    if density >= 0.70:
        return "High"
    if density >= 0.40:
        return "Medium"
    return "Low"


def encode_signal(signal: str) -> int:
    return {"RED": 0, "YELLOW": 1, "GREEN": 2}.get(signal, 0)


if not DATA_FILE.exists():
    st.error("Traffic data file is missing.")
    st.code("python simulation.py")
    st.stop()

if not MODEL_FILE.exists():
    st.error("Trained model is missing.")
    st.code("python train_model.py")
    st.stop()

try:
    data = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)
    model = load_model(str(MODEL_FILE), MODEL_FILE.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load the project data/model: {exc}")
    st.info("Run `python simulation.py` and then `python train_model.py`.")
    st.stop()

if data.empty:
    st.error("The traffic dataset is empty. Regenerate it using simulation.py.")
    st.stop()

st.sidebar.header("Simulation controls")
city_mode = st.sidebar.selectbox(
    "Traffic scenario", ["Normal", "Rush Hour", "Emergency Vehicle"]
)
step = st.sidebar.slider("Simulation time step", 0, len(data) - 1, 0)
show_map = st.sidebar.checkbox("Show illustrative map", value=True)
st.sidebar.caption("Changing the scenario changes the selected simulated snapshot.")

record = data.iloc[step].copy()
density = float(record["Traffic_Density"])
vehicle_count = int(record["Vehicle_Count"])
speed = float(record["Average_Speed"])
emergency = bool(int(record["Emergency_Vehicle"]))

if city_mode == "Rush Hour":
    density = min(1.0, density * 1.30)
    vehicle_count = int(vehicle_count * 1.20)
    speed = max(5.0, speed * 0.80)
elif city_mode == "Emergency Vehicle":
    emergency = True
    density = min(1.0, density * 0.90)

signal = signal_for(density, emergency)
congestion = congestion_for(density)

left, right = st.columns([1, 3])
with left:
    st.subheader("Current scenario")
    st.metric("Vehicle count", f"{vehicle_count:,}")
    st.metric("Traffic density", f"{density:.1%}")
    st.metric("Average speed", f"{speed:.1f} km/h")
    st.metric("Signal status", signal)
    st.metric("Congestion", congestion)
with right:
    st.subheader("Traffic density over time")
    chart_data = data.iloc[max(0, step - 71): step + 1][["Time", "Traffic_Density"]].copy()
    if not chart_data.empty:
        chart_data.loc[chart_data.index[-1], "Traffic_Density"] = density
    fig = px.line(
        chart_data,
        x="Time",
        y="Traffic_Density",
        title="Recent simulated density",
        labels={"Traffic_Density": "Density", "Time": "Time"},
    )
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Alerts and V2X status")
if emergency:
    st.warning("Emergency vehicle scenario active: signal is set to GREEN in this simulation.")
elif density >= 0.70:
    st.warning("High congestion: consider an adaptive signal timing strategy.")
elif density >= 0.40:
    st.info("Medium congestion: continue monitoring traffic density.")
else:
    st.success("Low congestion in the selected simulated snapshot.")

st.write(
    f"**V2X status:** simulated vehicle-to-infrastructure (V2I) message "
    f"{'priority enabled' if emergency else 'normal priority'}."
)
st.caption(
    "No real vehicle, roadside unit, GPS feed, or traffic signal is connected."
)

# Prepare one-step-ahead model input in the same shape as training.
time_value = pd.to_datetime(record["Time"]).timestamp()
model_input = pd.DataFrame(
    [[
        time_value,
        vehicle_count,
        density,
        speed,
        int(emergency),
        encode_signal(signal),
    ]],
    columns=FEATURES,
)
try:
    predicted_density = float(np.clip(model.predict(model_input)[0], 0, 1))
    st.subheader("Machine-learning prediction")
    p1, p2 = st.columns(2)
    p1.metric("Predicted next-step density", f"{predicted_density:.1%}")
    p2.metric("Change from current density", f"{predicted_density - density:+.1%}")
    st.caption("Prediction is generated from synthetic data; it is not a real-time forecast.")
except Exception as exc:
    st.error(f"Prediction failed: {exc}")

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.subheader("Congestion distribution")
    counts = data["Congestion_Level"].value_counts().reindex(
        ["Low", "Medium", "High"], fill_value=0
    )
    pie = px.pie(
        names=counts.index,
        values=counts.values,
        hole=0.35,
        title="Across the generated dataset",
    )
    st.plotly_chart(pie, use_container_width=True)
with c2:
    st.subheader("Vehicle count vs. average speed")
    scatter = px.scatter(
        data.iloc[::10],
        x="Vehicle_Count",
        y="Average_Speed",
        color="Congestion_Level",
        title="Sampled synthetic records",
        labels={"Vehicle_Count": "Vehicle count", "Average_Speed": "Speed (km/h)"},
    )
    st.plotly_chart(scatter, use_container_width=True)

st.divider()
st.subheader("Illustrative intersection view")
fig2 = go.Figure()
lanes = [
    ([-2, 2], [0, 0]),
    ([0, 0], [-2, 2]),
]
for x, y in lanes:
    fig2.add_trace(go.Scatter(x=x, y=y, mode="lines", line={"width": 28, "color": "#555"}, showlegend=False))
fig2.add_trace(
    go.Scatter(
        x=[-1.3, -0.7, 0.7, 1.3],
        y=[0.25, 0.25, -0.25, -0.25],
        mode="markers",
        marker={"size": 15, "color": ["#f39c12", "#3498db", "#e74c3c", "#2ecc71"]},
        name="Simulated vehicles",
    )
)
fig2.update_layout(
    title="Concept view only — vehicle positions are not live",
    xaxis={"visible": False, "range": [-2.5, 2.5]},
    yaxis={"visible": False, "range": [-2.5, 2.5], "scaleanchor": "x"},
    height=400,
)
st.plotly_chart(fig2, use_container_width=True)

if show_map:
    st.subheader("Illustrative route map")
    try:
        import folium
        from streamlit_folium import st_folium

        # A fixed demo location is used; it does not represent the user's location.
        map_view = folium.Map(location=[12.9716, 77.5946], zoom_start=13)
        folium.Marker(
            [12.9716, 77.5946],
            tooltip="Simulated intersection",
            popup=f"Density: {density:.1%} | Signal: {signal}",
            icon=folium.Icon(color="red" if congestion == "High" else "blue"),
        ).add_to(map_view)
        folium.Circle(
            [12.9716, 77.5946],
            radius=250 + density * 1000,
            color="red" if congestion == "High" else "blue",
            fill=True,
            fill_opacity=0.15,
        ).add_to(map_view)
        st_folium(map_view, width=1200, height=450, returned_objects=[])
        st.caption("Fixed example map near Bengaluru; marker and circle are illustrative.")
    except ImportError:
        st.info(
            "Map packages are not installed. Install `folium` and `streamlit-folium` "
            "to display the illustrative map."
        )
    except Exception as exc:
        st.warning(f"Map could not be displayed: {exc}")

st.divider()
st.subheader("Traffic data preview")
st.dataframe(data.iloc[max(0, step - 9):step + 1], use_container_width=True)
st.download_button(
    "Download generated traffic data (CSV)",
    data=data.to_csv(index=False).encode("utf-8"),
    file_name="traffic_data.csv",
    mime="text/csv",
)

with st.expander("Important project limitations"):
    st.markdown(
        """
        - The dataset is synthetic and generated locally.
        - V2X messages, emergency priority, and map overlays are simulated.
        - The model predicts the next row's density, not actual future city traffic.
        - This dashboard is for learning and portfolio demonstration only.
        """
    )
