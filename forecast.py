import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
"""
Energy‑price forecasting – streamlined version
---------------------------------------------
* **Targets:** forward %-change for each horizon in ``WINDOWS``.
* **Features:** auto‑generated lags, rolling mean/std, and historical %-changes
  for the same set of windows, plus cyclical hour & weekday encodings.
* **Outlier clipping removed** as requested – only NaNs/±Inf are discarded.
* **Plots removed** – script now prints metrics only.

Run with: ``python forecast.py`` (Python ≥3.9; GPU optional).
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0 — configuration & constants
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH    = Path("price_timeseries.csv")        # assumes script & CSV same dir
WINDOWS      = [1, 3, 5, 15, 30, 60, 90, 120, 168]  # hours ahead / behind
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# XGBoost settings (GPU optional – switch tree_method to 'hist' for CPU‑only)
BASE_MODEL = XGBRegressor(
    tree_method   = "gpu_hist",
    predictor     = "gpu_predictor",
    gpu_id        = 0,
    n_estimators  = 300,
    learning_rate = 0.05,
    max_depth     = 6,
    random_state  = RANDOM_STATE,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1 — load & resample price series
# ──────────────────────────────────────────────────────────────────────────────
print("↳ Loading", DATA_PATH)
df = (
    pd.read_csv(DATA_PATH, parse_dates=["datetime"])
      .assign(price=lambda d: pd.to_numeric(d["price"], errors="coerce"))
      .set_index("datetime")
      .resample("h")          # lowercase alias avoids FutureWarning
      .mean()
)
print("  Loaded rows after resample:", len(df))

df["price"] = df["price"].interpolate()

# ──────────────────────────────────────────────────────────────────────────────
# 2 — cyclical encodings
# ──────────────────────────────────────────────────────────────────────────────
df["hour"]   = df.index.hour
df["dow"]    = df.index.dayofweek
TWO_PI = 2 * np.pi
df["sin_hr"] = np.sin(TWO_PI * df["hour"] / 24)
df["cos_hr"] = np.cos(TWO_PI * df["hour"] / 24)
df["sin_dw"] = np.sin(TWO_PI * df["dow"]  / 7)
df["cos_dw"] = np.cos(TWO_PI * df["dow"]  / 7)

# ──────────────────────────────────────────────────────────────────────────────
# 3 — window‑based features
# ──────────────────────────────────────────────────────────────────────────────
for w in WINDOWS:
    df[f"lag_{w}"]        = df["price"].shift(w)
    df[f"roll_mean_{w}"]  = df["price"].rolling(w, min_periods=1).mean()
    df[f"roll_std_{w}"]   = df["price"].rolling(w, min_periods=1).std(ddof=0) if w > 1 else 0.0
    df[f"pct_change_{w}"] = df["price"].pct_change(w)

# ──────────────────────────────────────────────────────────────────────────────
# 4 — forward targets (future %-change)
# ──────────────────────────────────────────────────────────────────────────────
for h in WINDOWS:
    df[f"target_{h}"] = (df["price"].shift(-h) - df["price"]) / df["price"]

# ──────────────────────────────────────────────────────────────────────────────
# 5 — cleaning (NaNs / ±Inf only)
# ──────────────────────────────────────────────────────────────────────────────
print("↳ Rows before NaN drop:", len(df))
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print("  Rows after NaN drop:", len(df))

if df.empty:
    raise ValueError("All rows were dropped after NaN removal – please check the CSV.")

# ──────────────────────────────────────────────────────────────────────────────
# 6 — feature / target matrices
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    *[c for c in df.columns if c.startswith(("lag_", "roll_mean_", "roll_std_", "pct_change_"))],
    "sin_hr", "cos_hr", "sin_dw", "cos_dw",
]
TARGET_COLS = [f"target_{h}" for h in WINDOWS]

X, y = df[FEATURE_COLS], df[TARGET_COLS]
print("↳ Final feature matrix:", X.shape)

# ──────────────────────────────────────────────────────────────────────────────
# 7 — chronological train / test split
# ──────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=False
)
print("  Train rows:", len(X_train), "| Test rows:", len(X_test))

# ──────────────────────────────────────────────────────────────────────────────
# 8 — model fit
# ──────────────────────────────────────────────────────────────────────────────
print("↳ Fitting XGBoost …")
model = MultiOutputRegressor(BASE_MODEL, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ──────────────────────────────────────────────────────────────────────────────
# 9 — evaluation metrics (prints only)
# ──────────────────────────────────────────────────────────────────────────────
print("\n───────────  BACK‑TEST  ───────────")
for i, h in enumerate(WINDOWS):
    mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    r2  = r2_score        (y_test.iloc[:, i], y_pred[:, i])
    print(f"H{h:>3}:  MSE={mse:.6e}  R²={r2:+.3f}")
print("──────────────────────────────────")
print("Global R²:", r2_score(y_test, y_pred, multioutput="uniform_average"))

print("✓ Done.")
print("──────────────────────────────────")
import joblib

# Save the trained model to a file
MODEL_PATH = Path("xgb_multihorizon_model.pkl")
joblib.dump(model, MODEL_PATH)
print(f"✓ Model saved to {MODEL_PATH}")