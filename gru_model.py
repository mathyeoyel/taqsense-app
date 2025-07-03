import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam

# === Load CSV ===
df = pd.read_csv("ssd-rainfall-adm2-full.csv", low_memory=False)

# Fix date column
df.rename(columns={"#date": "date"}, inplace=True)
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["date"])
df = df.sort_values("date").reset_index(drop=True)


# === Process one region for training ===
region_name = "Juba"  # You can change this to another ADM2_NAME
region_df = df[df["ADM2_NAME"] == region_name].copy()
region_df["rfh"] = pd.to_numeric(region_df["rfh"], errors="coerce").fillna(method="ffill")

# === Normalize ===
scaler = MinMaxScaler()
region_df["rfh_scaled"] = scaler.fit_transform(region_df[["rfh"]])

# === Prepare sequences ===
data = region_df["rfh_scaled"].values
window_size = 30
X, y = [], []

for i in range(window_size, len(data)):
    X.append(data[i - window_size:i])
    y.append(data[i])

X = np.array(X)
y = np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# === Train-Test Split ===
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]

# === Build and Train GRU Model ===
model = Sequential([
    GRU(64, input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer=Adam(0.001), loss="mean_squared_error")  # ✅ Full loss name
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=1)

# === Save model and scaler ===
model.save("taqsense_gru_model.h5")
import joblib
joblib.dump(scaler, "taqsense_scaler.save")
