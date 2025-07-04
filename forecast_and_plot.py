import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load model and scaler from disk

def load_model_and_scaler(model_path='taqsense_gru_model.h5', scaler_path='taqsense_scaler.save'):
    """
    Loads the trained GRU model and scaler from disk.
    Returns (model, scaler).
    """
    model = load_model(model_path)
    import joblib
    scaler = joblib.load(scaler_path)
    return model, scaler

# Prepare sequences for forecasting

def prepare_sequences(series, window_size=30):
    """
    Takes a 1D pandas Series or array and returns a numpy array
    of shape (n_sequences, window_size) for forecasting.
    """
    arr = series.values if isinstance(series, pd.Series) else np.array(series)
    seqs = []
    for i in range(window_size, len(arr)):
        seqs.append(arr[i-window_size:i])
    return np.array(seqs)

# Rolling forecast utility

def rolling_forecast(series, model, scaler, window_size=30):
    """
    Performs a rolling one-step forecast over the entire series.
    Returns a DataFrame with 'date', 'actual', and 'predicted' columns.
    """
    seqs = prepare_sequences(series, window_size)
    preds = []
    for seq in seqs:
        scaled = scaler.transform(seq.reshape(-1,1)).reshape(1,window_size,1)
        pred_scaled = model.predict(scaled)
        pred = scaler.inverse_transform(pred_scaled)[0,0]
        preds.append(pred)
    dates = series.index[window_size:]
    df = pd.DataFrame({'date': dates,
                       'actual': series.values[window_size:],
                       'predicted': preds})
    return df
