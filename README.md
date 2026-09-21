# 🚦 AI-Driven V2X Smart Traffic Coordination System

A Streamlit dashboard that simulates a smart city intersection using
V2X (Vehicle-to-Everything) concepts. It generates synthetic traffic data,
applies rule-based signal control, and includes a small ML model for
next-step traffic density prediction.

> Built as part of my B.E. Electrical and Electronics Engineering final-year
> work and extended as a data science portfolio project.

<!-- Add a screenshot here: ![Dashboard](screenshot.png) -->

## Features

- Live-style dashboard (vehicle count, density, speed, signal status)
- Traffic modes: Normal, Rush Hour, Emergency
- Rule-based smart signal logic (emergency vehicles get priority green)
- Congestion and accident-risk indicators
- V2V network graph, 3D traffic view, congestion heatmap
- Interactive city map (Folium)
- Random Forest model predicting the next time step's traffic density

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Streamlit, Plotly, Folium

## Project Structure

```
├── app.py            # Streamlit dashboard
├── simulation.py     # Generates synthetic traffic data -> data/traffic_data.csv
├── train_model.py    # Trains the Random Forest -> models/traffic_model.pkl
├── data/             # Generated dataset
├── models/           # Saved model
└── requirements.txt
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Regenerate the data and retrain the model
python simulation.py
python train_model.py

# 3. Launch the dashboard
streamlit run app.py
```

Note: the dashboard advances one time step every 2 minutes
(`REFRESH_TIME_SECONDS` in `app.py`). Lower it for a faster demo.

## Model Results (honest note)

The data is **simulated**: vehicle count is random at every step, so the next
step's density is essentially unpredictable. Results on a chronological
80/20 split:

| Model | MAE |
|---|---|
| Baseline (always predict the average) | 0.2403 |
| Random Forest | 0.2512 |

The model does not beat the baseline, which is expected for random data.
The ML part demonstrates the pipeline (feature encoding, time-ordered
split, baseline comparison, model saving and loading in an app), not a
production-ready predictor.

An earlier version of `train_model.py` used the current density as both a
feature and the target (data leakage), which gave a misleadingly near-zero
error. That was fixed by predicting the *next* step's density instead.

## Future Improvements

- Simulate realistic traffic patterns (rush-hour peaks, time-of-day trends)
  so there is a real signal to learn
- Try time-series models (lags, rolling averages, ARIMA / LSTM)
- Use real traffic datasets
- Connect the MQTT/CoAP V2X simulation from the final-year project

## Author

Vrshy
