import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the GRU model
model = load_model("taqsense_gru_model.h5", compile=False)
model.compile(optimizer="adam", loss="mean_squared_error")  # recompile

# Load your preprocessed test data
df = pd.read_csv("ssd-rainfall-adm2-full.csv")

# Optional: Filter for one region (e.g., Juba)
df = df[df["ADM2_PCODE"] == "SS0101"]  # Replace with your desired region code

# Convert date and sort
df["date"] = pd.to_datetime(df["date"], errors='coerce')
df = df.dropna(subset=["date"])
df = df.sort_values("date")

# Use rfh (rainfall estimate) as target feature
data = df["rfh"].astype(float).fillna(method='ffill').values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Prepare test input for prediction
X = []
look_back = 5
for i in range(look_back, len(data_scaled)):
    X.append(data_scaled[i - look_back:i])

X = np.array(X)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Predict
predicted_scaled = model.predict(X)
predicted = scaler.inverse_transform(predicted_scaled)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df["date"].values[look_back:], data[look_back:], label="Actual Rainfall")
plt.plot(df["date"].values[look_back:], predicted, label="Predicted Rainfall")
plt.title("TaqSense Rainfall Forecast (GRU Model)")
plt.xlabel("Date")
plt.ylabel("Rainfall Estimate (RFH)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
