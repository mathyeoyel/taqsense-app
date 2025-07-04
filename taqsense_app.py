import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import pydeck as pdk
import forecast_and_plot as fp

# Default filename
DEFAULT_CSV = 'ssd-rainfall-with-coordinates.csv'

# Sidebar: CSV upload (optional)
uploaded_file = st.sidebar.file_uploader(
    "Upload rainfall CSV (columns: date, ADM2_NAME, rfh_avg) or leave blank to auto-detect CSV in app folder:",
    type=['csv']
)

# Helper: find CSV in current dir
def find_default_csv(default_name):
    if os.path.exists(default_name):
        return default_name
    for f in os.listdir('.'):
        if f.lower().endswith('.csv'):
            return f
    return None

# Load dataset
def load_data(uploaded):
    try:
        if uploaded is not None:
            df = pd.read_csv(uploaded, comment='#')
        else:
            csv_file = find_default_csv(DEFAULT_CSV)
            if csv_file is None:
                st.error("No CSV found. Upload a file or add one to the app directory.")
                st.stop()
            df = pd.read_csv(csv_file, comment='#')
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df.rename(columns={'ADM2_NAME': 'region', 'rfh_avg': 'rainfall'}, inplace=True)
        if df['date'].isna().any():
            st.error("Some dates could not be parsed. Check CSV date formats.")
            st.stop()
        return df[['date', 'region', 'rainfall']]
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

# Sequence helper
def prepare_sequences(series, window_size):
    arr = series.values
    return np.array([arr[i-window_size:i] for i in range(window_size, len(arr))])

# Load data, model, scaler
data = load_data(uploaded_file)
model = fp.model
scaler = fp.scaler

# ─── Interactive Map for Region Selection ───
# Load full coords CSV and prepare lat/lon
coords_csv = find_default_csv(DEFAULT_CSV)
coords_df = pd.read_csv(coords_csv, comment='#')
coords_df['date'] = pd.to_datetime(coords_df['date'], dayfirst=True, errors='coerce')
coords_df.rename(columns={'ADM2_NAME':'region','Latitude':'lat','Longitude':'lon'}, inplace=True)
coords_df = coords_df.dropna(subset=['lat','lon']).drop_duplicates(subset=['region'])

st.sidebar.markdown("### Or select region via map👇")

# Initialize session state for map selection
if 'map_selected' not in st.session_state:
    st.session_state.map_selected = None

layer = pdk.Layer(
    "ScatterplotLayer",
    data=coords_df,
    get_position='[lon, lat]',
    pickable=True,
    get_radius=20000,
    get_fill_color=[0, 128, 255, 160],
)

view_state = pdk.ViewState(
    latitude=float(coords_df['lat'].mean()),
    longitude=float(coords_df['lon'].mean()),
    zoom=5,
    pitch=0,
)

r = st.sidebar.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=view_state,
    layers=[layer],
    tooltip={"text": "{region}"}
))

# Capture clicks
if r.selected_rows:
    st.session_state.map_selected = r.selected_rows[0]['region']

# Health check endpoint
params = st.query_params
if "health" in params:
    st.write("healthy")
    st.stop()

# Optional page configuration
st.set_page_config(
    page_title="Taqsense Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Taqsense Rainfall Forecasting & Backtest Dashboard")

# Date-range picker
min_date = data['date'].min().date()
max_date = data['date'].max().date()
start_date, end_date = st.sidebar.date_input(
    "Select date range:",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)
if start_date > end_date:
    st.sidebar.error("End date must fall after start date.")

# Mode selector
tool = st.sidebar.selectbox("Mode", ["Forecast", "Backtest", "Tune Window Size"])

# Region selector (uses map click as default if available)
regions = sorted(data['region'].unique())
if tool == "Forecast":
    default = [st.session_state.map_selected] if st.session_state.map_selected else [regions[0]]
    selected_regions = st.sidebar.multiselect(
        "Select region(s) for forecasting:", regions, default=default
    )
else:
    default_idx = (
        regions.index(st.session_state.map_selected)
        if st.session_state.map_selected in regions else 0
    )
    selected_region = st.sidebar.selectbox(
        "Select region:", regions, index=default_idx
    )

# Hyperparameter sliders
window_size = st.sidebar.slider("Window size (dekads)", 10, 60, 30, 10)

# --------- Forecast Branch ---------
if tool == "Forecast":
    st.header("Multi-Region Forecast")
    out_frames = []
    for region in selected_regions:
        series = (
            data[(data['region'] == region) &
                 (data['date'].dt.date >= start_date) &
                 (data['date'].dt.date <= end_date)]
            .set_index('date')['rainfall']
            .asfreq('10D')
        )
        seqs = prepare_sequences(series.fillna(method='ffill').fillna(method='bfill'), window_size)
        if seqs.size == 0:
            st.warning(f"Not enough data for {region}.")
            continue
        last_seq = seqs[-1].reshape(1, window_size, 1)
        preds = scaler.inverse_transform(model.predict(last_seq)).flatten()
        future_dates = [series.index[-1] + datetime.timedelta(days=10*(i+1)) for i in range(len(preds))]
        df_pred = pd.DataFrame({
            'date': future_dates,
            'rainfall': preds,
            'type': 'predicted',
            'region': region
        })
        df_actual = pd.DataFrame({
            'date': series.index,
            'rainfall': series.values,
            'type': 'actual',
            'region': region
        })
        out_frames.append(pd.concat([df_actual, df_pred], ignore_index=True))
    if out_frames:
        df_all = pd.concat(out_frames, ignore_index=True)
        fig = px.line(
            df_all, x='date', y='rainfall', color='region', line_dash='type',
            labels={'rainfall':'Rainfall','line_dash':'Series Type'}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            "Download Forecast Results CSV",
            df_all.to_csv(index=False),
            "multi_region_forecast.csv"
        )

# --------- Backtest Branch ---------
elif tool == "Backtest":
    st.header("Backtest Analysis")
    series = (
        data[(data['region'] == selected_region) &
             (data['date'].dt.date >= start_date) &
             (data['date'].dt.date <= end_date)]
        .set_index('date')['rainfall']
        .asfreq('10D')
    )
    filled = series.fillna(method='ffill').fillna(method='bfill')
    seqs = prepare_sequences(filled, window_size)
    if seqs.size == 0:
        st.warning("Not enough data for backtest.")
    else:
        # batch predict
        all_preds = model.predict(seqs, batch_size=64).flatten()
        all_preds = scaler.inverse_transform(all_preds.reshape(-1,1)).flatten()
        y_true = filled.values[window_size:]
        dates = filled.index[window_size:]
        df_bt = pd.DataFrame({
            'date': dates,
            'actual': y_true,
            'predicted': all_preds
        })
        # Metrics
        mae = mean_absolute_error(y_true, all_preds)
        mse = mean_squared_error(y_true, all_preds)
        r2 = r2_score(y_true, all_preds)
        st.metric("MAE", f"{mae:.2f}")
        st.metric("MSE", f"{mse:.2f}")
        st.metric("R²", f"{r2:.2f}")
        # Plot
        df_melt = df_bt.melt(id_vars='date', value_vars=['actual','predicted'], var_name='type', value_name='rainfall')
        fig2 = px.line(df_melt, x='date', y='rainfall', color='type', labels={'rainfall':'Rainfall','type':'Series'})
        st.plotly_chart(fig2, use_container_width=True)
        st.download_button(
            "Download Backtest CSV",
            df_bt.to_csv(index=False),
            f"backtest_{selected_region}.csv"
        )

# --------- Tuning Branch ---------
elif tool == "Tune Window Size":
    st.header("Window Size Tuning")
    steps = list(range(10, 61, 10))
    results = []
    for w in steps:
        seqs = prepare_sequences(
            data[data['region']==selected_region]
            .set_index('date')['rainfall']
            .asfreq('10D')
            .fillna(method='ffill').fillna(method='bfill'),
            w
        )
        if seqs.size == 0:
            continue
        preds = model.predict(seqs, batch_size=64).flatten()
        preds = scaler.inverse_transform(preds.reshape(-1,1)).flatten()
        y_true = data[data['region']==selected_region].set_index('date')['rainfall'].asfreq('10D').fillna(method='ffill').fillna(method='bfill').values[w:]
        mae = mean_absolute_error(y_true, preds)
        results.append({'window_size': w, 'MAE': mae})
    df_tune = pd.DataFrame(results)
    best = df_tune.loc[df_tune['MAE'].idxmin()]
    st.write(f"**Best window size:** {int(best.window_size)} with MAE={best.MAE:.2f}")
    st.dataframe(df_tune.set_index('window_size'))

# -----------------------------
# Packaging & Deployment Guide
# -----------------------------
# (Dockerfile, requirements.txt, Render guide, etc.)
