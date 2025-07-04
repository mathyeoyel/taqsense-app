import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import pydeck as pdk
import forecast_and_plot as fp  # must expose `model` & `scaler` at top level

# ── Helpers & Config ──

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
        if uploaded:
            df = pd.read_csv(uploaded, comment='#')
        else:
            path = find_default_csv(DEFAULT_CSV)
            if not path:
                st.error("No CSV found. Upload one or add `ssd-rainfall-with-coordinates.csv` to the app folder.")
                st.stop()
            df = pd.read_csv(path, comment='#')
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df.rename(columns={'ADM2_NAME':'region','rfh_avg':'rainfall'}, inplace=True)
        if df['date'].isna().any():
            st.error("Some dates could not be parsed. Check your `date` column format.")
            st.stop()
        return df[['date','region','rainfall']]
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

def prepare_sequences(series, window_size):
    arr = series.values
    seqs = [arr[i-window_size:i] for i in range(window_size, len(arr))]
    return np.array(seqs).reshape(-1, window_size, 1)

# ── Load Data & Model ──

data = load_data(st.sidebar.file_uploader(
    "Upload rainfall CSV (columns: date, ADM2_NAME, rfh_avg) or leave blank to auto-detect",
    type=['csv'])
)
model  = fp.model
scaler = fp.scaler

# ── Interactive Map for Region Selection ──

# Load coords CSV
coords_path = find_default_csv(DEFAULT_CSV)
coords_df   = pd.read_csv(coords_path, comment='#')

def parse_coord(s):
    if pd.isna(s): return np.nan
    m = re.match(r'([0-9\.]+)°\s*([NSEW])', s.strip())
    if not m: return np.nan
    val, hemi = m.groups()
    val = float(val)
    return -val if hemi in ('S','W') else val

coords_df.rename(columns={
    'ADM2_NAME':'region',
    'Latitude':'lat_str',
    'Longitude':'lon_str'
}, inplace=True)
coords_df['lat'] = coords_df['lat_str'].apply(parse_coord)
coords_df['lon'] = coords_df['lon_str'].apply(parse_coord)
coords_df = coords_df.dropna(subset=['lat','lon']).drop_duplicates('region')

st.sidebar.markdown("### Or select region via map👇")
if 'map_selected' not in st.session_state:
    st.session_state.map_selected = None

layer = pdk.Layer(
    "ScatterplotLayer",
    data=coords_df,
    get_position='[lon, lat]',
    pickable=True,
    get_radius=20000,
    get_fill_color=[0,128,255,160]
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
r = st.sidebar.pydeck_chart(deck)

selected = r.selected_rows()
if selected:
    st.session_state.map_selected = selected[0]['region']

# ── Health Check & Page Setup ──

params = st.query_params
if "health" in params:
    st.write("healthy")
    st.stop()

st.set_page_config(
    page_title="Taqsense Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Taqsense Rainfall Forecasting & Backtest Dashboard")

# ── Sidebar Controls ──

# Date range
min_d = data['date'].min().date()
max_d = data['date'].max().date()
start_d, end_d = st.sidebar.date_input(
    "Select date range:", [min_d, max_d], min_value=min_d, max_value=max_d
)
if start_d > end_d:
    st.sidebar.error("End date must fall after start date.")

# Mode
tool = st.sidebar.selectbox("Mode", ["Forecast","Backtest","Tune Window Size"])

# Region(s)
regions = sorted(data['region'].unique())
if tool=="Forecast":
    default = [st.session_state.map_selected] if st.session_state.map_selected else [regions[0]]
    selected_regions = st.sidebar.multiselect("Select region(s) for forecasting:", regions, default=default)
else:
    idx = regions.index(st.session_state.map_selected) if st.session_state.map_selected in regions else 0
    selected_region = st.sidebar.selectbox("Select region:", regions, index=idx)

# Hyperparameter
window_size = st.sidebar.slider("Window size (dekads)", 10, 60, 30, 10)

# ── Forecast Branch ──

if tool=="Forecast":
    st.header("Multi-Region Forecast")
    out = []
    for reg in selected_regions:
        series = (
            data[(data['region']==reg)]
            .set_index('date')['rainfall']
            .asfreq('10D')
        )
        seqs = prepare_sequences(series.fillna(method='ffill').fillna(method='bfill'), window_size)
        if seqs.size==0:
            st.warning(f"Not enough data for {reg}")
            continue
        last = seqs[-1].reshape(1,window_size,1)
        preds = scaler.inverse_transform(model.predict(last)).flatten()
        future = [series.index[-1]+datetime.timedelta(days=10*(i+1)) for i in range(len(preds))]
        df_pred = pd.DataFrame({'date':future,'rainfall':preds,'type':'predicted','region':reg})
        df_act  = pd.DataFrame({'date':series.index,'rainfall':series.values,'type':'actual','region':reg})
        out.append(pd.concat([df_act,df_pred],ignore_index=True))
    if out:
        df_all = pd.concat(out,ignore_index=True)
        fig = px.line(df_all, x='date',y='rainfall',color='region',line_dash='type',
                      labels={'rainfall':'Rainfall','line_dash':'Series Type'})
        st.plotly_chart(fig,use_container_width=True)
        st.download_button("Download Forecast CSV", df_all.to_csv(index=False), "forecast.csv")

# ── Backtest Branch ──

elif tool=="Backtest":
    st.header("Backtest Analysis")
    series = (
        data[(data['region']==selected_region)]
        .set_index('date')['rainfall']
        .asfreq('10D')
    )
    filled = series.fillna(method='ffill').fillna(method='bfill')
    seqs = prepare_sequences(filled, window_size)
    if seqs.size==0:
        st.warning("Not enough data for backtest.")
    else:
        preds = model.predict(seqs, batch_size=64).flatten()
        preds = scaler.inverse_transform(preds.reshape(-1,1)).flatten()
        true = filled.values[window_size:]
        dates = filled.index[window_size:]
        df_bt = pd.DataFrame({'date':dates,'actual':true,'predicted':preds})
        mae = mean_absolute_error(true,preds)
        mse = mean_squared_error(true,preds)
        r2  = r2_score(true,preds)
        st.metric("MAE",f"{mae:.2f}")
        st.metric("MSE",f"{mse:.2f}")
        st.metric("R²",f"{r2:.2f}")
        dfm = df_bt.melt(id_vars='date',value_vars=['actual','predicted'],var_name='type',value_name='rainfall')
        fig2 = px.line(dfm,x='date',y='rainfall',color='type',labels={'rainfall':'Rainfall','type':'Series'})
        st.plotly_chart(fig2,use_container_width=True)
        st.download_button("Download Backtest CSV", df_bt.to_csv(index=False), f"backtest_{selected_region}.csv")

# ── Tuning Branch ──

elif tool=="Tune Window Size":
    st.header("Window Size Tuning")
    series_full = (
        data[data['region']==selected_region]
        .set_index('date')['rainfall']
        .asfreq('10D').fillna(method='ffill').fillna(method='bfill')
    )
    steps = list(range(10,61,10))
    res = []
    for w in steps:
        seqs = prepare_sequences(series_full, w)
        if seqs.size==0: continue
        p = model.predict(seqs, batch_size=64).flatten()
        p = scaler.inverse_transform(p.reshape(-1,1)).flatten()
        y = series_full.values[w:]
        res.append({'window_size':w,'MAE':mean_absolute_error(y,p)})
    df_t = pd.DataFrame(res).set_index('window_size')
    best = df_t['MAE'].idxmin()
    st.write(f"**Best window size:** {best} (MAE={df_t.loc[best,'MAE']:.2f})")
    st.dataframe(df_t)

# ── Packaging & Deployment Guide ──
# (Your Dockerfile, requirements.txt, and Render instructions here)
