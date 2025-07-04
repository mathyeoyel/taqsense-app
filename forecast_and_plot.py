# forecast_and_plot.py

import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError

# Locate the saved model and scaler
BASE_DIR    = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(BASE_DIR, "taqsense_gru_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "taqsense_scaler.save")

def load_model_and_scaler():
    """
    Load and return:
      - model: a Keras GRU model loaded with custom_objects for 'mse'
      - scaler: a sklearn MinMaxScaler loaded from SCALER_PATH
    """
    model = load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"mse": MeanSquaredError()}
    )
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def prepare_sequences(series: pd.Series, window_size: int) -> np.ndarray:
    arr = series.values
    seqs = []
    for i in range(window_size, len(arr)):
        seqs.append(arr[i-window_size : i])
    return np.array(seqs).reshape(-1, window_size, 1)

def rolling_forecast(series: pd.Series, window_size: int) -> pd.DataFrame:
    s = series.asfreq("10D").fillna(method="ffill").fillna(method="bfill")
    dates = s.index[window_size:]
    seqs  = prepare_sequences(s, window_size)
    preds_scaled = model.predict(seqs, batch_size=64).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
    actual = s.values[window_size:]
    return pd.DataFrame({"date": dates, "actual": actual, "predicted": preds})

# Immediately load and expose
model, scaler = load_model_and_scaler()
