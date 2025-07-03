"""
retrain_model.py

This script retrains your GRU model on the full dataset and
saves the updated model and scaler to disk.
Schedule this via Render Cron (e.g. daily at midnight).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense
from sklearn.metrics import mean_squared_error

def load_and_prepare(path):
    df = pd.read_csv(path, comment='#')
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df.rename(columns={'ADM2_NAME':'region', 'rfh_avg':'rainfall'}, inplace=True)
    # Example for a single region (Juba) or loop for all
    df_juba = df[df['region']=='Juba'].set_index('date')['rainfall'].asfreq('10D').ffill().bfill()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_juba.values.reshape(-1,1))
    # create sequences
    window = 30
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i,0])
        y.append(scaled[i,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(-1, window, 1)
    return X, y, scaler

if __name__ == '__main__':
    path = 'ssd-rainfall-with-coordinates.csv'
    X, y, scaler = load_and_prepare(path)

    # Build model
    model = Sequential([
        GRU(64, input_shape=(X.shape[1],1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train
    model.fit(X, y, epochs=20, batch_size=32)

    # Save
    model.save('taqsense_gru_model.h5')
    import joblib
    joblib.dump(scaler, 'taqsense_scaler.save')

    print("Retraining complete and models saved.")