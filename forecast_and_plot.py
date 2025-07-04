# forecast_and_plot.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os

# Paths to your saved model and scaler
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "taqsense_gru_model.h5")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "taqsense_scaler.save")

def load_model_and_scaler():
    """
    Load the trained GRU model and the MinMaxScaler.
    Returns:
        model  -- a Keras model loaded from MODEL_PATH
        scaler -- a sklearn MinMaxScaler loaded from SCALER_PATH
    """
    model  = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def prepare_sequences(series: pd.Series, window_size: int) -> np.ndarray:
    """
    Convert a 1D Pandas Series into a 3D numpy array of sliding windows
    suitable for (num_samples, window_size, 1) input to the GRU.
    """
    arr = series.values
    sequences = []
    for i in range(window_size, len(arr)):
        sequences.append(arr[i-window_size : i])
    return np.array(sequences).reshape(-1, window_size, 1)

def rolling_forecast(series: pd.Series, model, scaler, window_size: int) -> pd.DataFrame:
    """
    Perform a rolling 1-step forecast over the entire series:
    - slides a window of length window_size over the series
    - at each step, predicts the next value
    - inverse-transforms predictions to original scale
    Returns:
        DataFrame with columns ['date', 'actual', 'predicted']
    """
    # Ensure series has a DatetimeIndex
    series = series.asfreq('10D').fillna(method='ffill').fillna(method='bfill')
    dates = series.index[window_size:]
    seqs  = prepare_sequences(series, window_size)
    # Batch predict
    preds_scaled = model.predict(seqs, batch_size=64).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
    actual = series.values[window_size:]
    return pd.DataFrame({
        "date":      dates,
        "actual":    actual,
        "predicted": preds
    })

# Expose model and scaler at module level for easy import
model, scaler = load_model_and_scaler()
