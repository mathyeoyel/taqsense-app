import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
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

# Streamlit UI
# Health check endpoint
params = st.experimental_get_query_params()
if "health" in params:
    st.write("healthy")
    st.stop()

# Optional page configuration
st.set_page_config(
    page_title="Taqsense Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title = "Taqsense Rainfall Forecasting & Backtest Dashboard"

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
tool = st.sidebar.selectbox("Mode", ["Forecast", "Backtest", "Tune Window Size"] )

# Region selector
regions = sorted(data['region'].unique())
if tool == "Forecast":
    selected_regions = st.sidebar.multiselect(
        "Select region(s) for forecasting:", regions, default=[regions[0]]
    )
else:
    selected_region = st.sidebar.selectbox("Select region:", regions)

# Hyperparameter sliders
window_size = st.sidebar.slider("Window size (dekads)", 10, 60, 30, 10)

# Forecast branch
if tool == "Forecast":
    st.header("Multi-Region Forecast")
    out_frames = []
    for region in selected_regions:
        # filter series
        series = (
            data[(data['region']==region) &
                 (data['date'].dt.date>=start_date) &
                 (data['date'].dt.date<=end_date)]
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
        st.plotly_chart(fig)
        st.download_button(
            "Download Forecast Results CSV",
            df_all.to_csv(index=False),
            "multi_region_forecast.csv"
        )

# Backtest and Tune remain unchanged...

# -----------------------------
# Packaging & Deployment
# -----------------------------
# To containerize this Streamlit app, create the following Dockerfile in the app directory:
#
# --- Dockerfile ---
# FROM python:3.10-slim
# WORKDIR /app
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . .
# EXPOSE 8501
# CMD ["streamlit", "run", "taqsense_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# --- end Dockerfile ---
#
# And a requirements.txt with pinned dependencies:
#
# --- requirements.txt ---
# streamlit>=1.20.0
# pandas
# numpy
# scikit-learn
# plotly
# tensorflow  # or keras, depending on model
#
# To build and run the container locally:
#   docker build -t taqsense-app .
#   docker run -p 8501:8501 taqsense-app
#
# For deployment to Streamlit Community Cloud:
# 1. Push this repo to GitHub
# 2. In Streamlit Cloud, create a new app pointing to this repo and branch
# 3. Specify the command `streamlit run taqsense_app.py`
# 4. Cloud will auto-install from requirements.txt and launch your app.

# Render Deployment Guide
# ----------------------
# 1. Create a GitHub repository and push all app files (including Dockerfile & requirements.txt).
# 2. Sign up or log in to Render (https://render.com) and create a new Web Service.
#    - Connect your GitHub account and select your repo.
#    - For Environment, choose "Docker" (Render will auto-detect your Dockerfile).
#    - Set the start command to:
#        streamlit run taqsense_app.py --server.port=$PORT
#    - The default port variable $PORT is provided by Render.
# 3. In the "Advanced" settings:
#    - Add any Environment Variables (e.g., STREAMLIT_SERVER_HEADLESS=true).
#    - Adjust instance type if needed (512MB RAM free tier is default).
# 4. Click "Create Web Service". Render will build and deploy your container.
# 5. Once live, Render provides a URL (e.g., https://taqsense-app.onrender.com).
# 6. To update, simply push new commits to GitHub; Render auto-deploys.
#
# Optional: Scheduled Retraining
# - In Render Dashboard, create a new Cron Job service.
# - Use the same repo, set the command to run a retraining script, e.g.: 
#       python retrain_model.py
# - Configure the schedule (e.g., daily at midnight) via cron syntax.

# End of Render Deployment Guide
# End of taqsense_app.py
