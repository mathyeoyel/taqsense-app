import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import pydeck as pdk
import forecast_and_plot as fp  # must expose `model` & `scaler`

# ── Page Configuration ──
st.set_page_config(
    page_title="Taqsense Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Health Check ──
params = st.query_params
if "health" in params:
    st.write("healthy")
    st.stop()

# ── Helpers ──
DEFAULT_CSV = 'ssd-rainfall-with-coordinates.csv'

def find_default_csv(default_name):
    if os.path.exists(default_name):
        return default_name
    for f in os.listdir('.'):
        if f.lower().endswith('.csv'):
            return f
    return None

def load_data(uploaded):
    try:
        if uploaded is not None:
            df = pd.read_csv(uploaded, comment='#')
        else:
            path = find_default_csv(DEFAULT_CSV)
            if not path:
                st.sidebar.error("No CSV found. Upload one or add the CSV to the app folder.")
                st.stop()
            df = pd.read_csv(path, comment='#')
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df.rename(columns={'ADM2_NAME':'region','rfh_avg':'rainfall'}, inplace=True)
        if df['date'].isna().any():
            st.sidebar.error("Some dates could not be parsed. Check your CSV date formats.")
            st.stop()
        return df[['date','region','rainfall']]
    except Exception as e:
        st.sidebar.error(f"Error loading CSV: {e}")
        st.stop()

def prepare_sequences(series: pd.Series, window_size: int) -> np.ndarray:
    arr = series.values
    seqs = [arr[i-window_size:i] for i in range(window_size, len(arr))]
    return np.array(seqs).reshape(-1, window_size, 1)

def parse_coord(s):
    """Extract float from strings like '4.6000° N' or '33.1234° W'."""
    if pd.isna(s):
        return np.nan
    s = s.strip()
    m = re.match(r'([0-9\.]+)°\s*([NSEW])', s)
    if not m:
        return np.nan
    val, hemi = m.groups()
    val = float(val)
    return -val if hemi in ('S','W') else val

# ── Load Data & Model ──
uploaded_file = st.sidebar.file_uploader(
    "Upload rainfall CSV (date, ADM2_NAME, rfh_avg) or leave blank to auto-detect:",
    type=['csv']
)
data = load_data(uploaded_file)
model  = fp.model
scaler = fp.scaler

# ── Interactive Map (visual only) ──
coords_path = find_default_csv(DEFAULT_CSV)
coords_df   = pd.read_csv(coords_path, comment='#')
coords_df.rename(columns={
    'ADM2_NAME':'region',
    'Latitude':'lat_str',
    'Longitude':'lon_str'
}, inplace=True)
coords_df['lat'] = coords_df['lat_str'].apply(parse_coord)
coords_df['lon'] = coords_df['lon_str'].apply(parse_coord)
coords_df = coords_df.dropna(subset=['lat','lon']).drop_duplicates('region')

st.sidebar.markdown("### Map of Counties (visual only)👇")
layer = pdk.Layer(
    "ScatterplotLayer",
    data=coords_df,
    get_position='[lon, lat]',
    pickable=False,
    get_radius=20000,
    get_fill_color=[0,128,255,160],
)
view = pdk.ViewState(
    latitude=float(coords_df['lat'].mean()),
    longitude=float(coords_df['lon'].mean()),
    zoom=5, pitch=0
)
deck = pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=view,
    layers=[layer],
    tooltip={"text":"{region}"}
)
st.sidebar.pydeck_chart(deck)

# ── Sidebar Controls ──

# Date-range picker
min_d = data['date'].min().date()
max_d = data['date'].max().date()
start_d, end_d = st.sidebar.date_input(
    "Select date range:",
    [min_d, max_d],
    min_value=min_d,
    max_value=max_d
)
if start_d > end_d:
    st.sidebar.error("End date must be on or after the start date.")

# Mode selector
tool = st.sidebar.selectbox("Mode", ["Forecast", "Backtest", "Tune Window Size"])

# Region selector
regions = sorted(data['region'].unique())
if tool == "Forecast":
    selected_regions = st.sidebar.multiselect(
        "Select region(s) for forecasting:", regions, default=[regions[0]]
    )
else:
    selected_region = st.sidebar.selectbox("Select region:", regions)

# Hyperparameter slider
window_size = st.sidebar.slider("Window size (dekads)", 10, 60, 30, 10)

# ── Main ──
st.title("Taqsense Rainfall Forecasting & Backtest Dashboard")

if tool == "Forecast":
    st.header("Multi-Region Forecast")
    out_frames = []
    for reg in selected_regions:
        series = (
            data[(data['region']==reg) & 
                 (data['date'].dt.date>=start_d) & 
                 (data['date'].dt.date<=end_d)]
            .set_index('date')['rainfall']
            .asfreq('10D')
        )
        seqs = prepare_sequences(
            series.fillna(method='ffill').fillna(method='bfill'),
            window_size
        )
        if seqs.size == 0:
            st.warning(f"Not enough data for {reg}.")
            continue
        last_seq = seqs[-1].reshape(1, window_size, 1)
        preds = scaler.inverse_transform(model.predict(last_seq)).flatten()
        future_dates = [
            series.index[-1] + datetime.timedelta(days=10*(i+1))
            for i in range(len(preds))
        ]
        df_pred = pd.DataFrame({
            'date': future_dates,
            'rainfall': preds,
            'type': 'predicted',
            'region': reg
        })
        df_act = pd.DataFrame({
            'date': series.index,
            'rainfall': series.values,
            'type': 'actual',
            'region': reg
        })
        out_frames.append(pd.concat([df_act, df_pred], ignore_index=True))
    if out_frames:
        df_all = pd.concat(out_frames, ignore_index=True)
        fig = px.line(
            df_all, x='date', y='rainfall',
            color='region', line_dash='type',
            labels={'rainfall':'Rainfall','line_dash':'Series Type'}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            "Download Forecast CSV",
            df_all.to_csv(index=False),
            "forecast.csv"
        )

elif tool == "Backtest":
    st.header("Backtest Analysis")
    series = (
        data[(data['region']==selected_region) & 
             (data['date'].dt.date>=start_d) & 
             (data['date'].dt.date<=end_d)]
        .set_index('date')['rainfall']
        .asfreq('10D')
    )
    filled = series.fillna(method='ffill').fillna(method='bfill')
    seqs = prepare_sequences(filled, window_size)
    if seqs.size == 0:
        st.warning("Not enough data for backtest.")
    else:
        preds = model.predict(seqs, batch_size=64).flatten()
        preds = scaler.inverse_transform(preds.reshape(-1,1)).flatten()
        y_true = filled.values[window_size:]
        dates = filled.index[window_size:]
        df_bt = pd.DataFrame({
            'date': dates,
            'actual': y_true,
            'predicted': preds
        })
        mae = mean_absolute_error(y_true, preds)
        mse = mean_squared_error(y_true, preds)
        r2  = r2_score(y_true, preds)
        st.metric("MAE", f"{mae:.2f}")
        st.metric("MSE", f"{mse:.2f}")
        st.metric("R²", f"{r2:.2f}")
        dfm = df_bt.melt(
            id_vars='date',
            value_vars=['actual','predicted'],
            var_name='type',
            value_name='rainfall'
        )
        fig2 = px.line(
            dfm, x='date', y='rainfall',
            color='type',
            labels={'rainfall':'Rainfall','type':'Series'}
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.download_button(
            "Download Backtest CSV",
            df_bt.to_csv(index=False),
            f"backtest_{selected_region}.csv"
        )

elif tool == "Tune Window Size":
    st.header("Window Size Tuning")
    series_full = (
        data[data['region']==selected_region]
        .set_index('date')['rainfall']
        .asfreq('10D')
        .fillna(method='ffill').fillna(method='bfill')
    )
    steps = list(range(10, 61, 10))
    results = []
    for w in steps:
        seqs = prepare_sequences(series_full, w)
        if seqs.size == 0:
            continue
        preds = model.predict(seqs, batch_size=64).flatten()
        preds = scaler.inverse_transform(preds.reshape(-1,1)).flatten()
        y_true = series_full.values[w:]
        mae = mean_absolute_error(y_true, preds)
        results.append({'window_size': w, 'MAE': mae})
    df_tune = pd.DataFrame(results).set_index('window_size')
    best = df_tune['MAE'].idxmin()
    st.write(f"**Best window size:** {best} (MAE={df_tune.loc[best,'MAE']:.2f})")
    st.dataframe(df_tune)

# -----------------------------
# Packaging & Deployment Guide
# -----------------------------
# (Dockerfile, requirements.txt, Render instructions…)
