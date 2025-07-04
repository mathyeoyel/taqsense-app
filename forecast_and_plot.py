import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Paths (adjust filenames if yours differ)
BASE_DIR    = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(BASE_DIR, "taqsense_gru_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "taqsense_scaler.save")

def load_model_and_scaler():
    """
    Load and return:
      - model: a Keras GRU model loaded from MODEL_PATH
      - scaler: a sklearn MinMaxScaler loaded from SCALER_PATH
    """
    model  = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def prepare_sequences(series: pd.Series, window_size: int) -> np.ndarray:
    arr = series.values
    seqs = []
    for i in range(window_size, len(arr)):
        seqs.append(arr[i-window_size : i])
    return np.array(seqs).reshape(-1, window_size, 1)

def rolling_forecast(series: pd.Series, window_size: int) -> pd.DataFrame:
    # Ensure dekadal frequency and no gaps
    s = series.asfreq('10D').fillna(method='ffill').fillna(method='bfill')
    dates = s.index[window_size:]
    seqs  = prepare_sequences(s, window_size)
    preds_scaled = model.predict(seqs, batch_size=64).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
    actual = s.values[window_size:]
    return pd.DataFrame({"date": dates, "actual": actual, "predicted": preds})

# Load once at import-time so fp.model and fp.scaler exist
model, scaler = load_model_and_scaler()
