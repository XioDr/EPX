import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# ──────────────────────────────────────────────────────────────────────────────
# Configuration & constants (should match your training script)
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH    = Path("price_timeseries.csv")        # Your original CSV
MODEL_PATH   = Path("xgb_multihorizon_model.pkl")
WINDOWS      = [1, 3, 5, 15, 30, 60, 90, 120, 168]

# ──────────────────────────────────────────────────────────────────────────────
# 1 — Load trained model
# ──────────────────────────────────────────────────────────────────────────────
print(f"Loading trained model from {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# 2 — Load & preprocess latest price series
# ──────────────────────────────────────────────────────────────────────────────
print(f"Reading data from {DATA_PATH}")
df = (
    pd.read_csv(DATA_PATH, parse_dates=["datetime"])  
      .assign(price=lambda d: pd.to_numeric(d["price"], errors="coerce"))
      .set_index("datetime")
      .resample("h").mean()
)
df["price"] = df["price"].interpolate()

# Cyclical encodings
TWO_PI = 2 * np.pi
df["hour"] = df.index.hour
df["dow"]  = df.index.dayofweek
df["sin_hr"] = np.sin(TWO_PI * df["hour"] / 24)
df["cos_hr"] = np.cos(TWO_PI * df["hour"] / 24)
df["sin_dw"] = np.sin(TWO_PI * df["dow"]  / 7)
df["cos_dw"] = np.cos(TWO_PI * df["dow"]  / 7)

# Window‑based features
for w in WINDOWS:
    df[f"lag_{w}"]        = df["price"].shift(w)
    df[f"roll_mean_{w}"]  = df["price"].rolling(w, min_periods=1).mean()
    df[f"roll_std_{w}"]   = df["price"].rolling(w, min_periods=1).std(ddof=0) if w > 1 else 0.0
    df[f"pct_change_{w}"] = df["price"].pct_change(w)

# ──────────────────────────────────────────────────────────────────────────────
# 3 — Prepare feature vector for prediction
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    *[c for c in df.columns if c.startswith(("lag_", "roll_mean_", "roll_std_", "pct_change_"))],
    "sin_hr", "cos_hr", "sin_dw", "cos_dw",
]
# Use the most recent timestamp's features
latest = df.iloc[-1]
X_pred = latest[FEATURE_COLS].values.reshape(1, -1)

# ──────────────────────────────────────────────────────────────────────────────
# 4 — Make predictions
# ──────────────────────────────────────────────────────────────────────────────
print("Generating forecasts for the next horizons:")
y_pct = model.predict(X_pred)[0]
current_price = latest["price"]
predicted = {h: current_price * (1 + dy) for h, dy in zip(WINDOWS, y_pct)}

# Display results
for horizon, price in predicted.items():
    print(f"{horizon:>3}h ahead: {price:.4f}")

print("Done.")
