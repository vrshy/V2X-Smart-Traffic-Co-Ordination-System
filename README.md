# V2X Smart Traffic Coordination System

A beginner-friendly Python portfolio demo that generates synthetic traffic data,
trains a next-step traffic-density regression model, and displays results in Streamlit.

## Files

- `simulation.py` — generates `data/traffic_data.csv`.
- `train_model.py` — evaluates baseline predictions and trains a Random Forest model.
- `app.py` — launches the interactive Streamlit dashboard.
- `requirements.txt` — Python dependencies.

## Setup (Windows PowerShell)

Run these commands from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python simulation.py
python train_model.py
streamlit run app.py
```

If PowerShell blocks activation, use the virtual-environment Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe simulation.py
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Notes

All traffic data, V2X messages, vehicle positions, and map overlays are illustrative.
This project does not connect to live traffic systems and must not be used to control
real traffic signals. Check model MAE against the baselines printed by `train_model.py`;
lower MAE is better.
