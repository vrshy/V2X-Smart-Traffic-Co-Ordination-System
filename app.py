import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
import joblib
import folium
from streamlit_folium import st_folium

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="V2X Smart Traffic System",
    layout="wide"
)

# ---------------------------------------------------
# CACHE DATA
# ---------------------------------------------------

@st.cache_data
def load_data():

    return pd.read_csv(
        "data/traffic_data.csv"
    )

df = load_data()

# ---------------------------------------------------
# CACHE MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():

    return joblib.load(
        "models/traffic_model.pkl"
    )

model = load_model()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("⚙ Traffic Controls")

city_mode = st.sidebar.selectbox(
    "City Traffic Mode",
    ["Normal", "Rush Hour", "Emergency"]
)

# ---------------------------------------------------
# TRAFFIC MODE SETTINGS
# ---------------------------------------------------

density_multiplier = 1.0

if city_mode == "Rush Hour":

    density_multiplier = 1.4

elif city_mode == "Emergency":

    density_multiplier = 0.8

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title(
    "🚦 AI-Driven V2X Smart Traffic Coordination System"
)

st.markdown(
    f"### Intelligent Transportation Analytics • Mode: `{city_mode}`"
)

st.info(
    "🔄 Dashboard refreshes every 2 minutes"
)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "frame_index" not in st.session_state:

    st.session_state.frame_index = 10

i = st.session_state.frame_index

# ---------------------------------------------------
# CURRENT FRAME
# ---------------------------------------------------

current_df = df.iloc[:i].copy()

current_df["Traffic_Density"] = (
    current_df["Traffic_Density"] *
    density_multiplier
)

latest = current_df.iloc[-1]

vehicle_count = int(
    latest["Vehicle_Count"]
)

traffic_density = float(
    latest["Traffic_Density"]
)

signal_status = latest["Signal_Status"]

emergency_vehicle = int(
    latest["Emergency_Vehicle"]
)

average_speed = float(
    latest["Average_Speed"]
)

congestion_level = latest["Congestion_Level"]

# ---------------------------------------------------
# SMART SIGNAL LOGIC
# ---------------------------------------------------

if emergency_vehicle == 1:

    signal_status = "GREEN"

elif traffic_density > 0.7:

    signal_status = "GREEN"

elif traffic_density > 0.4:

    signal_status = "YELLOW"

else:

    signal_status = "RED"

# ---------------------------------------------------
# AI TRAFFIC PREDICTION
# ---------------------------------------------------

signal_map = {
    "RED": 0,
    "YELLOW": 1,
    "GREEN": 2
}

prediction_input = [[
    vehicle_count,
    traffic_density,
    average_speed,
    emergency_vehicle,
    signal_map[signal_status]
]]

predicted_density = model.predict(
    prediction_input
)[0]

predicted_density = round(
    float(predicted_density),
    2
)

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "🚗 Vehicles",
    vehicle_count
)

c2.metric(
    "📊 Density",
    f"{traffic_density:.2f}"
)

c3.metric(
    "🚘 Avg Speed",
    f"{average_speed:.1f} km/h"
)

c4.metric(
    "🚦 Signal",
    signal_status
)

# ---------------------------------------------------
# ALERTS
# ---------------------------------------------------

if emergency_vehicle == 1:

    st.error(
        "🚑 Emergency Vehicle Detected"
    )

    st.success(
        "🚑 Smart Emergency Route Activated"
    )

    st.warning(
        "📢 Emergency Broadcast Sent "
        "to Nearby Vehicles"
    )

# Congestion Alerts
if congestion_level == "HIGH":

    st.warning(
        "⚠ High Traffic Congestion"
    )

elif congestion_level == "MEDIUM":

    st.info(
        "⚠ Moderate Traffic Flow"
    )

else:

    st.success(
        "✅ Smooth Traffic Flow"
    )

# ---------------------------------------------------
# V2X STATUS
# ---------------------------------------------------

st.subheader(
    "📡 V2X Communication"
)

st.success(
    "Vehicle-to-Vehicle and "
    "Vehicle-to-Infrastructure "
    "Communication Active"
)

# ---------------------------------------------------
# AI PREDICTION
# ---------------------------------------------------

st.subheader(
    "🤖 AI Traffic Prediction"
)

st.info(
    f"Predicted Next-Step Traffic Density: "
    f"{predicted_density}"
)

# ---------------------------------------------------
# CHARTS
# ---------------------------------------------------

r1c1, r1c2 = st.columns(2)

# Vehicle Count
with r1c1:

    fig_vehicle = px.line(
        current_df,
        x="Time",
        y="Vehicle_Count",
        title="Vehicle Count Analysis"
    )

    st.plotly_chart(
        fig_vehicle,
        use_container_width=True,
        key=f"vehicle_chart_{i}"
    )

# Density
with r1c2:

    fig_density = px.area(
        current_df,
        x="Time",
        y="Traffic_Density",
        title="Traffic Density"
    )

    st.plotly_chart(
        fig_density,
        use_container_width=True,
        key=f"density_chart_{i}"
    )

# ---------------------------------------------------
# SPEED + SIGNAL
# ---------------------------------------------------

r2c1, r2c2 = st.columns(2)

with r2c1:

    fig_speed = px.line(
        current_df,
        x="Time",
        y="Average_Speed",
        title="Average Traffic Speed"
    )

    st.plotly_chart(
        fig_speed,
        use_container_width=True,
        key=f"speed_chart_{i}"
    )

with r2c2:

    signal_counts = (
        current_df["Signal_Status"]
        .value_counts()
        .reset_index()
    )

    signal_counts.columns = [
        "Signal",
        "Count"
    ]

    fig_signal = px.pie(
        signal_counts,
        names="Signal",
        values="Count",
        title="Signal Distribution"
    )

    st.plotly_chart(
        fig_signal,
        use_container_width=True,
        key=f"signal_chart_{i}"
    )

# ---------------------------------------------------
# 3D TRAFFIC VISUALIZATION
# ---------------------------------------------------

st.subheader(
    "🌐 3D Traffic Flow Simulation"
)

x = np.random.rand(vehicle_count) * 100
y = np.random.rand(vehicle_count) * 100
z = np.random.rand(vehicle_count) * 10

fig_3d = go.Figure()

fig_3d.add_trace(
    go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(size=5)
    )
)

fig_3d.update_layout(
    height=500
)

st.plotly_chart(
    fig_3d,
    use_container_width=True,
    key=f"3d_chart_{i}"
)

# ---------------------------------------------------
# V2V NETWORK
# ---------------------------------------------------

st.subheader(
    "📡 Vehicle-to-Vehicle Communication"
)

num_nodes = min(
    vehicle_count,
    25
)

x_nodes = np.random.rand(num_nodes) * 100
y_nodes = np.random.rand(num_nodes) * 100

network_fig = go.Figure()

network_fig.add_trace(
    go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode='markers+text',
        text=[
            f"V{j}"
            for j in range(num_nodes)
        ],
        marker=dict(size=14)
    )
)

for a in range(num_nodes):

    for b in range(a + 1, num_nodes):

        dist = np.sqrt(
            (x_nodes[a] - x_nodes[b])**2 +
            (y_nodes[a] - y_nodes[b])**2
        )

        if dist < 30:

            network_fig.add_trace(
                go.Scatter(
                    x=[
                        x_nodes[a],
                        x_nodes[b]
                    ],
                    y=[
                        y_nodes[a],
                        y_nodes[b]
                    ],
                    mode='lines',
                    showlegend=False
                )
            )

st.plotly_chart(
    network_fig,
    use_container_width=True,
    key=f"network_chart_{i}"
)

# ---------------------------------------------------
# SMART INTERSECTION
# ---------------------------------------------------

st.subheader(
    "🚦 Smart Traffic Intersection"
)

intersection_fig = go.Figure()

intersection_fig.add_shape(
    type="rect",
    x0=0,
    y0=40,
    x1=100,
    y1=60
)

intersection_fig.add_shape(
    type="rect",
    x0=40,
    y0=0,
    x1=60,
    y1=100
)

num_cars = min(
    vehicle_count,
    20
)

car_x = np.random.uniform(
    0,
    100,
    num_cars
)

car_y = np.random.uniform(
    0,
    100,
    num_cars
)

intersection_fig.add_trace(
    go.Scatter(
        x=car_x,
        y=car_y,
        mode='markers',
        marker=dict(size=12)
    )
)

signal_color = "red"

if signal_status == "GREEN":

    signal_color = "green"

elif signal_status == "YELLOW":

    signal_color = "yellow"

intersection_fig.add_trace(
    go.Scatter(
        x=[50],
        y=[50],
        mode='markers',
        marker=dict(
            size=30,
            color=signal_color
        )
    )
)

st.plotly_chart(
    intersection_fig,
    use_container_width=True,
    key=f"intersection_chart_{i}"
)

# ---------------------------------------------------
# ACCIDENT RISK
# ---------------------------------------------------

st.subheader(
    "⚠ AI Accident Risk Detection"
)

accident_risk = "LOW"
risk_score = 20

if (
    traffic_density > 0.75 and
    average_speed > 60
):

    accident_risk = "HIGH"
    risk_score = 90

elif traffic_density > 0.5:

    accident_risk = "MEDIUM"
    risk_score = 60

risk_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk_score,
    title={
        'text': "Accident Risk Score"
    },
    gauge={
        'axis': {
            'range': [0, 100]
        }
    }
))

st.plotly_chart(
    risk_gauge,
    use_container_width=True,
    key=f"risk_chart_{i}"
)

# ---------------------------------------------------
# HEATMAP
# ---------------------------------------------------

st.subheader(
    "🔥 Traffic Congestion Heatmap"
)

heat_x = np.random.rand(vehicle_count) * 100
heat_y = np.random.rand(vehicle_count) * 100

heatmap_fig = go.Figure(
    data=go.Histogram2dContour(
        x=heat_x,
        y=heat_y,
        contours=dict(
            coloring='heatmap'
        )
    )
)

st.plotly_chart(
    heatmap_fig,
    use_container_width=True,
    key=f"heatmap_chart_{i}"
)

# ---------------------------------------------------
# EMERGENCY TRACKING
# ---------------------------------------------------

st.subheader(
    "🚑 Emergency Vehicle Tracking"
)

tracking_fig = go.Figure()

tracking_fig.add_trace(
    go.Scatter(
        x=np.linspace(0, 100, 50),
        y=np.linspace(0, 100, 50),
        mode='lines'
    )
)

if emergency_vehicle == 1:

    emergency_x = (i * 2) % 100
    emergency_y = (i * 2) % 100

    tracking_fig.add_trace(
        go.Scatter(
            x=[emergency_x],
            y=[emergency_y],
            mode='markers+text',
            text=["🚑"]
        )
    )

st.plotly_chart(
    tracking_fig,
    use_container_width=True,
    key=f"tracking_chart_{i}"
)

# ---------------------------------------------------
# ROUTE OPTIMIZATION
# ---------------------------------------------------

st.subheader(
    "🛣 Dynamic Route Optimization"
)

route_fig = go.Figure()

route_fig.add_trace(
    go.Scatter(
        x=[0, 20, 40, 60, 80, 100],
        y=[0, 10, 20, 40, 60, 80],
        mode='lines+markers',
        name='Original Route'
    )
)

route_fig.add_trace(
    go.Scatter(
        x=[0, 15, 30, 55, 75, 100],
        y=[0, 20, 35, 55, 70, 90],
        mode='lines+markers',
        name='Optimized Route'
    )
)

st.plotly_chart(
    route_fig,
    use_container_width=True,
    key=f"route_chart_{i}"
)

# ---------------------------------------------------
# REAL MAP
# ---------------------------------------------------

st.subheader(
    "🗺 Smart City Traffic Map"
)

if i % 3 == 0:

    base_lat = 10.7905
    base_lon = 79.1378

    traffic_map = folium.Map(
        location=[
            base_lat,
            base_lon
        ],
        zoom_start=13,
        control_scale=True
    )

    num_map_vehicles = min(
        vehicle_count,
        20
    )

    for v in range(num_map_vehicles):

        lat_offset = np.random.uniform(
            -0.02,
            0.02
        )

        lon_offset = np.random.uniform(
            -0.02,
            0.02
        )

        vehicle_lat = (
            base_lat +
            lat_offset
        )

        vehicle_lon = (
            base_lon +
            lon_offset
        )

        marker_color = "green"

        if congestion_level == "HIGH":

            marker_color = "red"

        elif congestion_level == "MEDIUM":

            marker_color = "orange"

        folium.CircleMarker(
            location=[
                vehicle_lat,
                vehicle_lon
            ],
            radius=5,
            color=marker_color,
            fill=True,
            fill_opacity=0.7,
            popup=f"Vehicle {v}"
        ).add_to(traffic_map)

    if emergency_vehicle == 1:

        emergency_lat = (
            base_lat +
            np.random.uniform(
                -0.01,
                0.01
            )
        )

        emergency_lon = (
            base_lon +
            np.random.uniform(
                -0.01,
                0.01
            )
        )

        folium.Marker(
            location=[
                emergency_lat,
                emergency_lon
            ],
            popup="Emergency Vehicle",
            icon=folium.Icon(
                color="red",
                icon="info-sign"
            )
        ).add_to(traffic_map)

    st_folium(
        traffic_map,
        width=1200,
        height=450,
        key=f"traffic_map_{i}"
    )

else:

    st.info(
        "🛰 Updating Smart Traffic Map..."
    )

# ---------------------------------------------------
# CONGESTION GAUGE
# ---------------------------------------------------

st.subheader(
    "📊 Congestion Score"
)

congestion_score = min(
    traffic_density * 100,
    100
)

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=congestion_score,
    title={
        'text': "Traffic Congestion"
    },
    gauge={
        'axis': {
            'range': [0, 100]
        }
    }
))

st.plotly_chart(
    gauge,
    use_container_width=True,
    key=f"gauge_chart_{i}"
)

# ---------------------------------------------------
# LIVE TELEMETRY
# ---------------------------------------------------

st.subheader(
    "📄 Live Traffic Telemetry"
)

st.dataframe(
    current_df.tail(10),
    use_container_width=True
)

# ---------------------------------------------------
# 2 MINUTE REFRESH
# ---------------------------------------------------

REFRESH_TIME_SECONDS = 120

time.sleep(
    REFRESH_TIME_SECONDS
)

st.session_state.frame_index = (
    i + 1
) % len(df)

st.rerun()