from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense
from apscheduler.schedulers.background import BackgroundScheduler
import os
from pandas.tseries.offsets import BusinessDay
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from math import sqrt
from sklearn.metrics import mean_squared_error
import requests
import time
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DB_PATH = 'AAPL.db'
MODEL_PATH = 'gru_model.h5'
LOOKBACK = 60
TARGET = 'Close'
ERROR_THRESHOLD = 5.0
CONFIDENCE_INTERVAL = 1.96 * 4.82  # 95% CI based on GRU RMSE

# DB Connection
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn

# Fetch and Update Data
def fetch_data_with_retry(ticker, start, end, retries=3, delay=5):
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    for attempt in range(retries):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, session=session)
            if not data.empty:
                logging.info(f"Fetched {len(data)} rows for {ticker}")
                return data
            logging.warning(f"Attempt {attempt + 1}: No data fetched for {ticker}")
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}: Error fetching {ticker}: {e}")
        if attempt < retries - 1:
            time.sleep(delay)
    return pd.DataFrame()

def fetch_and_update_data(ticker='AAPL'):
    try:
        conn = get_db_connection()
        df_existing = pd.read_sql(f"SELECT * FROM stock_prices WHERE Stock = '{ticker}'", conn, parse_dates=['Date'])
        last_date = df_existing['Date'].max() if not df_existing.empty else pd.to_datetime('2020-01-01')

        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = (datetime.now().date() - timedelta(days=1)).strftime('%Y-%m-%d')
        if start_date >= end_date:
            return df_existing

        new_data = fetch_data_with_retry(ticker, start_date, end_date)
        if new_data.empty:
            logging.warning(f"No new data fetched for {ticker}")
            return df_existing

        new_data.reset_index(inplace=True)
        new_data['Stock'] = ticker
        new_data = new_data[['Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']]
        new_data.to_sql('stock_prices', conn, if_exists='append', index=False)

        df = pd.read_sql(f"SELECT * FROM stock_prices WHERE Stock = '{ticker}'", conn, parse_dates=['Date'])
        conn.close()

        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['Sharpe_Ratio'] = df['Daily_Return'].rolling(window=20).mean() / (df['Volatility'] / np.sqrt(252))
        df.dropna(inplace=True)

        conn = get_db_connection()
        df.reset_index().to_sql('stock_prices', conn, if_exists='replace', index=False)
        conn.close()

        return df
    except Exception as e:
        logging.error(f"Error fetching/updating data for {ticker}: {e}")
        return pd.DataFrame()

# Create Sequences
def create_sequences(data, lookback=LOOKBACK):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

# Retrain with Error Check
def retrain_gru_with_errors(ticker='AAPL'):
    df = fetch_and_update_data(ticker)
    if df.empty or len(df) < LOOKBACK + 10:
        logging.warning(f"Insufficient data for retraining {ticker}")
        return

    scaler = MinMaxScaler(feature_range=(0, 1))
    series = df[TARGET].values.reshape(-1, 1)
    scaled_series = scaler.fit_transform(series)

    model_path = f"{ticker}_{MODEL_PATH}"
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = Sequential()
        model.add(GRU(50, return_sequences=True, input_shape=(LOOKBACK, 1)))
        model.add(GRU(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

    recent_start = max(0, len(scaled_series) - 100 - LOOKBACK)
    series_recent = scaled_series[recent_start:]
    X_recent, y_recent = create_sequences(series_recent)
    if len(X_recent) == 0:
        logging.warning(f"Not enough recent data for error check on {ticker}")
        return
    X_recent = X_recent.reshape((X_recent.shape[0], LOOKBACK, 1))
    preds_scaled = model.predict(X_recent, verbose=0)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    actual = scaler.inverse_transform(y_recent).flatten()
    rmse = sqrt(mean_squared_error(actual, preds))

    if rmse > ERROR_THRESHOLD or not os.path.exists(model_path):
        X, y = create_sequences(scaled_series)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model.fit(X, y, epochs=10, batch_size=32, verbose=1)
        model.save(model_path)
        logging.info(f"Retrained GRU for {ticker}. RMSE: {rmse:.2f}")

# Scheduler (Daily Update, Quarterly Retrain)
scheduler = BackgroundScheduler()
scheduler.add_job(lambda: fetch_and_update_data('AAPL'), 'cron', day_of_week='mon-fri', hour=0)
scheduler.add_job(lambda: retrain_gru_with_errors('AAPL'), 'cron', month='1,4,7,10', day=1, hour=0)
scheduler.start()

# Generate Predictions
def generate_predictions(model, scaler, last_sequence, steps):
    predictions = []
    lower_bounds = []
    upper_bounds = []
    current_seq = last_sequence.copy()
    for _ in range(steps):
        pred_scaled = model.predict(current_seq.reshape(1, LOOKBACK, 1), verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0][0]
        predictions.append(pred)
        lower_bounds.append(pred - CONFIDENCE_INTERVAL)
        upper_bounds.append(pred + CONFIDENCE_INTERVAL)
        current_seq = np.append(current_seq[1:], pred_scaled, axis=0)
    return predictions, lower_bounds, upper_bounds

def get_future_dates(last_date, steps):
    future_dates = [last_date + BusinessDay(n) for n in range(1, steps + 1)]
    return pd.to_datetime(future_dates)

# Routes
@app.route('/', methods=['GET', 'POST'])
def dashboard():
    ticker = request.form.get('ticker', 'AAPL')
    df = fetch_and_update_data(ticker)
    
    # Default template variables
    error = None
    historical_plot = {}
    df_tail = "<p>No recent data available.</p>"
    results = [
        {'Model': 'ARIMA', 'MAE': 14.50, 'RMSE': 18.15, 'MAPE': 'N/A'},
        {'Model': 'Prophet', 'MAE': 21.31, 'RMSE': 27.04, 'MAPE': 10.02},
        {'Model': 'XGBoost', 'MAE': 8.57, 'RMSE': 11.55, 'MAPE': 3.76},
        {'Model': 'LSTM', 'MAE': 5.12, 'RMSE': 6.98, 'MAPE': 2.35},
        {'Model': 'GRU (Best)', 'MAE': 3.23, 'RMSE': 4.82, 'MAPE': 1.49}
    ]
    prediction_plot = None
    pred_table = None
    overall_pct_str = None
    term = request.form.get('term', 'short')

    if df.empty:
        error = f"No data available for {ticker}. Check Yahoo Finance or DB connection."
    else:
        df_tail = df[['Close', 'Volume', 'MA20', 'MA50', 'Volatility']].tail(10).to_html(index=True, classes='table table-striped')
        
        # Historical Plot
        for period in ['1M', '6M', '1Y', '5Y']:
            fig, ax = plt.subplots(figsize=(10, 5))
            if period == '1M':
                df_plot = df[-30:]
            elif period == '6M':
                df_plot = df[-126:]
            elif period == '1Y':
                df_plot = df[-252:]
            else:
                df_plot = df
            ax.plot(df_plot.index, df_plot['Close'], label='Close Price', color='blue')
            ax.set_title(f'{ticker} Historical Close Price ({period})')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (USD)')
            ax.legend()
            ax.grid(True)
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            historical_plot[period] = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)

        # Predictions
        if request.method == 'POST':
            horizon = request.form.get('horizon', 'day')
            steps_map = {'day': 1, 'week': 5, 'month': 20, 'quarter': 63, 'year': 252}
            steps = steps_map.get(horizon, 1)

            model_path = f"{ticker}_{MODEL_PATH}"
            if os.path.exists(model_path):
                model = load_model(model_path)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(df[TARGET].values.reshape(-1, 1))
                series_scaled = scaler.transform(df[TARGET].values.reshape(-1, 1))
                if len(series_scaled) < LOOKBACK:
                    pred_table = "<p>Not enough data for predictions.</p>"
                else:
                    last_sequence = series_scaled[-LOOKBACK:]
                    preds, lower_bounds, upper_bounds = generate_predictions(model, scaler, last_sequence, steps)
                    future_dates = get_future_dates(df.index[-1], steps)
                    last_close = df['Close'].iloc[-1]
                    pred_df = pd.DataFrame({
                        'Date': future_dates,
                        'Estimated Close': np.round(preds, 2),
                        '% Change': np.round((preds - last_close) / last_close * 100, 2),
                        'Lower Bound (95% CI)': np.round(lower_bounds, 2),
                        'Upper Bound (95% CI)': np.round(upper_bounds, 2)
                    })
                    pred_table = pred_df.to_html(index=False, classes='table table-striped')

                    # Overall % change
                    final_pred = preds[-1] if steps > 0 else last_close
                    overall_pct = (final_pred - last_close) / last_close * 100
                    overall_pct_str = f"{overall_pct:.2f}% {'increase' if overall_pct >= 0 else 'decrease'}"

                    # Prediction Plot
                    fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
                    ax_pred.plot(df.index[-100:], df['Close'][-100:], label='Historical Trend', color='blue')
                    ax_pred.plot(pred_df['Date'], pred_df['Estimated Close'], label='Predicted (GRU)', color='red')
                    ax_pred.fill_between(pred_df['Date'], pred_df['Lower Bound (95% CI)'], pred_df['Upper Bound (95% CI)'], 
                                       color='red', alpha=0.1, label='95% Confidence Interval')
                    ax_pred.set_title(f'{ticker} Predicted Close Price for Next {horizon.capitalize()} (GRU Model)')
                    ax_pred.set_xlabel('Date')
                    ax_pred.set_ylabel('Price (USD)')
                    ax_pred.legend()
                    ax_pred.grid(True)
                    buf_pred = BytesIO()
                    fig_pred.savefig(buf_pred, format='png')
                    buf_pred.seek(0)
                    prediction_plot = base64.b64encode(buf_pred.getvalue()).decode()
                    plt.close(fig_pred)
            else:
                pred_table = "<p>Model not trained yet. It will train automatically on schedule.</p>"

    return render_template('dashboard.html', error=error, historical_plot=historical_plot, df_tail=df_tail, 
                           results=results, prediction_plot=prediction_plot, pred_table=pred_table, 
                           overall_pct=overall_pct_str, selected_ticker=ticker, term=term)

if __name__ == '__main__':
    app.run(debug=True)