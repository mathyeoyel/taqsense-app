# forecast_and_plot.py

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Locate the saved model and scaler relative to this file
BASE_DIR    = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(BASE_DIR, "taqsense_gru_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "taqsense_scaler.save")

def load_model_and_scaler():
    """
    Load and return:
      - model: a Keras GRU model loaded from MODEL_PATH
      - scaler: a sklearn MinMaxScaler loaded from SCALER_PATH
    """
    model  = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def prepare_sequences(series: pd.Series, window_size: int) -> np.ndarray:
    """
    Convert a 1D Pandas Series into a 3D NumPy array of shape
    (num_samples, window_size, 1) for GRU input.
    """
    arr = series.values
    seqs = []
    for i in range(window_size, len(arr)):
        seqs.append(arr[i-window_size : i])
    return np.array(seqs).reshape(-1, window_size, 1)

def rolling_forecast(series: pd.Series, model, scaler, window_size: int) -> pd.DataFrame:
    """
    Slides a window of length `window_size` over `series`,
    predicting one step ahead each time, and returns
    a DataFrame with columns ['date','actual','predicted'].
    """
    # Resample to dekadal frequency and fill gaps
    series = series.asfreq('10D').fillna(method='ffill').fillna(method='bfill')
    dates = series.index[window_size:]
    seqs  = prepare_sequences(series, window_size)
    # Batch prediction
    preds_scaled = model.predict(seqs, batch_size=64).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
    actual = series.values[window_size:]
    return pd.DataFrame({
        "date":      dates,
        "actual":    actual,
        "predicted": preds
    })

# Immediately load and expose for easy import
model, scaler = load_model_and_scaler()
