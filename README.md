# EPX

Multi-horizon **energy price forecasting** pipeline for hourly electricity price series (EPEX-style data).

## Contents

| File | Description |
| --- | --- |
| `time_series_maker.py` | Merges daily `.xlsx` price files into one padded, filtered workbook with a consistent header. |
| `forecast.py` | Builds lag, rolling-mean/std and historical %-change features plus cyclical hour/weekday encodings, then trains a multi-output XGBoost model predicting forward %-change for horizons of 1 to 168 hours. Prints metrics. |
| `prediction.py` | Loads the trained model and produces forecasts from the most recent observations. |

## Requirements

Python 3.9+, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `joblib`, `openpyxl`. A GPU is optional; switch XGBoost to `tree_method="hist"` for CPU-only runs.

## Workflow

1. Put the source `.xlsx` files in the input directory configured in `time_series_maker.py` and run it to create the merged workbook; export the series as `price_timeseries.csv` with a `datetime` column.
2. `python forecast.py` trains the model and saves `xgb_multihorizon_model.pkl`.
3. `python prediction.py` loads it and prints the forecasts.

Data files and trained models are not included.
