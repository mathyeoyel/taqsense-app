import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# === Load & Prepare Juba Rainfall Data ===
df = pd.read_csv("ssd-rainfall-adm2-full.csv", low_memory=False)
juba_df = df[df['ADM2_PCODE'] == 'SS0205'].copy()
juba_df['date'] = pd.to_datetime(juba_df['date'])
juba_df = juba_df.sort_values('date').reset_index(drop=True)

# === Optional: Plot Rainfall Trends ===
plt.figure(figsize=(12, 5))
plt.plot(juba_df['date'], juba_df['rfh'], label='Dekadal Rainfall (mm)')
plt.plot(juba_df['date'], juba_df['rfq'], label='% of Normal (rfq)', linestyle='--', color='orange')
plt.title('Rainfall Trends in Juba County (1981–Present)')
plt.xlabel('Date')
plt.ylabel('Rainfall / % Normal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Normalize rfh ===
rain_data = juba_df[['date', 'rfh']].copy()
rain_data['rfh'] = rain_data['rfh'].fillna(method='ffill')
scaler = MinMaxScaler()
rain_data['rfh_scaled'] = scaler.fit_transform(rain_data[['rfh']])

# === Create Sequences for GRU ===
data = rain_data['rfh_scaled'].values
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
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

# === Confirm Shapes ===
print("Input shape:", X.shape)
print("Target shape:", y.shape)
