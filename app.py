from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import sqlite3
import os
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.optimizers import Adam
from apscheduler.schedulers.background import BackgroundScheduler
import matplotlib
matplotlib.use('Agg')  # no GUI backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from math import sqrt
from init_db import query_stock_data
from pandas.tseries.offsets import BusinessDay

# --- App Setup ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
DB_PATH = 'D:/AAPL_STOCK_APP/AAPL.db'
MODEL_PATH = 'gru_model.keras'
LOOKBACK = 60
TARGET = 'Close'
ERROR_THRESHOLD = 6.5
CONFIDENCE_INTERVAL = 1.96 * 6.268
LEARNING_RATE = 0.1

# --- RL Agent ---
class RLAgent:
    def __init__(self):
        self.q_table = {}
        self.state = None
        self.last_reward = 0

    def get_state(self, rmse):
        return f"rmse_{int(rmse * 10)}"

    def choose_action(self, state):
        return "adjust" if np.random.random() < 0.1 else "no_adjust"

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table.get((state, action), 0.0)
        self.q_table[(state, action)] = current_q + LEARNING_RATE * (
            reward + 0.9 * max(
                self.q_table.get((next_state, "adjust"), 0.0),
                self.q_table.get((next_state, "no_adjust"), 0.0)
            ) - current_q
        )

agent = RLAgent()

# --- Helper Functions ---
def fetch_and_clean_data(ticker='AAPL'):
    df = query_stock_data(ticker, DB_PATH)
    if df.empty:
        logging.error(f"No data available for {ticker} in {DB_PATH}")
        return pd.DataFrame()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    return df.dropna()
def create_sequences(data, lookback=LOOKBACK):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def train_or_update_model(df, model_path):
    if df.empty or len(df) < LOOKBACK + 10:
        logging.warning("Insufficient data for training")
        return None, {}

    scaler = MinMaxScaler(feature_range=(0, 1))
    series = df[TARGET].values.reshape(-1, 1)
    scaled_series = scaler.fit_transform(series)

    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = Sequential()
        model.add(Input(shape=(LOOKBACK, 1)))
        model.add(GRU(50, return_sequences=True))
        model.add(GRU(50))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    X, y = create_sequences(scaled_series)
    if len(X) == 0:
        return model, {}

    X = X.reshape((X.shape[0], LOOKBACK, 1))

    # --- RL Adjustment ---
    recent_start = max(0, len(scaled_series) - 100 - LOOKBACK)
    series_recent = scaled_series[recent_start:]
    X_recent, y_recent = create_sequences(series_recent)

    metrics = {}
    if len(X_recent) > 0:
        X_recent = X_recent.reshape((X_recent.shape[0], LOOKBACK, 1))
        preds_scaled = model.predict(X_recent, verbose=0)
        preds = scaler.inverse_transform(preds_scaled).flatten()
        actual = scaler.inverse_transform(y_recent).flatten()

        metrics["short_term"] = calculate_metrics(actual, preds)

        rmse = metrics["short_term"]["RMSE"]
        state = agent.get_state(rmse)
        action = agent.choose_action(state)
        reward = -rmse if rmse > ERROR_THRESHOLD else 1.0 / (rmse + 1e-5)
        next_state = agent.get_state(rmse * 0.9 if action == "adjust" else rmse)
        agent.update_q_value(state, action, reward, next_state)

        if action == "adjust" and rmse > ERROR_THRESHOLD:
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            logging.info(f"Model adjusted due to high RMSE: {rmse:.2f}")

        # Long-term evaluation (use last 252 days)
        long_start = max(0, len(scaled_series) - 252 - LOOKBACK)
        X_long, y_long = create_sequences(scaled_series[long_start:])
        if len(X_long) > 0:
            X_long = X_long.reshape((X_long.shape[0], LOOKBACK, 1))
            preds_long = scaler.inverse_transform(model.predict(X_long, verbose=0)).flatten()
            actual_long = scaler.inverse_transform(y_long).flatten()
            metrics["long_term"] = calculate_metrics(actual_long, preds_long)

    model.save(model_path)
    return model, metrics

def generate_predictions(model, scaler, last_sequence, steps, volatility):
    predictions, lower_bounds, upper_bounds = [], [], []
    current_seq = last_sequence.copy()
    for _ in range(steps):
        pred_scaled = model.predict(current_seq.reshape(1, LOOKBACK, 1), verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0][0]
        adjusted_pred = pred + (volatility * np.random.normal(0, 0.1))
        predictions.append(adjusted_pred)
        ci_adjustment = CONFIDENCE_INTERVAL * (volatility / 10)
        lower_bounds.append(adjusted_pred - ci_adjustment)
        upper_bounds.append(adjusted_pred + ci_adjustment)
        current_seq = np.append(current_seq[1:], pred_scaled, axis=0)
    return predictions, lower_bounds, upper_bounds

def get_future_dates(last_date, steps):
    return pd.to_datetime([last_date + BusinessDay(n) for n in range(1, steps + 1)])

def generate_trading_signal(last_close, final_pred, overall_pct, volatility, lower_bound, upper_bound):
    threshold = 2.0
    ci_width = (upper_bound - lower_bound) / last_close * 100
    confidence = "High" if ci_width < 2 else "Medium" if ci_width < 5 else "Low"

    if volatility > 0.03:
        return "HOLD", f"High volatility ({volatility:.2f}), staying neutral.", confidence

    if overall_pct > threshold:
        return "BUY", f"Predicted increase {overall_pct:.2f}% (from {last_close:.2f} to {final_pred:.2f}).", confidence
    elif overall_pct < -threshold:
        return "SELL", f"Predicted decrease {overall_pct:.2f}% (from {last_close:.2f} to {final_pred:.2f}).", confidence
    else:
        return "HOLD", f"Small predicted change ({overall_pct:.2f}%).", confidence

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def dashboard():
    ticker = request.form.get('ticker', 'AAPL')
    term = request.form.get('term', 'short')
    df = fetch_and_clean_data(ticker)

    error, metrics, historical_plot = None, {}, {}
    prediction_plot, pred_table, overall_pct_str = None, None, None
    trading_signal, signal_reason, signal_confidence = None, None, None

    if df.empty:
        error = f"No data for {ticker}."
    else:
        # Historical Plots
        for period, days in zip(['1M', '6M', '1Y', '5Y'], [30, 126, 252, len(df)]):
            fig, ax = plt.subplots(figsize=(10, 5))
            df_plot = df[-days:]
            ax.plot(df_plot.index, df_plot['Close'], label='Close Price', color='blue')
            ax.set_title(f'{ticker} Close Price ({period})')
            ax.legend(); ax.grid(True)
            buf = BytesIO(); fig.savefig(buf, format='png'); buf.seek(0)
            historical_plot[period] = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)

        # Train/update model
        model_path = f"{ticker}_{MODEL_PATH}"
        model, metrics = train_or_update_model(df, model_path)

        if request.method == 'POST' and model:
            horizon = request.form.get('horizon', 'day')
            steps_map = {'day': 1, 'week': 5, 'month': 20, 'quarter': 63, 'year': 252}
            steps = steps_map.get(horizon, 1)

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(df[TARGET].values.reshape(-1, 1))
            series_scaled = scaler.transform(df[TARGET].values.reshape(-1, 1))

            if len(series_scaled) >= LOOKBACK:
                last_sequence = series_scaled[-LOOKBACK:]
                recent_volatility = df['Volatility'].iloc[-1] if 'Volatility' in df.columns else 0.1
                preds, lower_bounds, upper_bounds = generate_predictions(
                    model, scaler, last_sequence, steps, recent_volatility
                )
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

                final_pred = preds[-1]
                overall_pct = (final_pred - last_close) / last_close * 100
                overall_pct_str = f"{overall_pct:.2f}% {'increase' if overall_pct >= 0 else 'decrease'}"

                trading_signal, signal_reason, signal_confidence = generate_trading_signal(
                    last_close, final_pred, overall_pct, recent_volatility,
                    lower_bounds[-1], upper_bounds[-1]
                )

                fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
                ax_pred.plot(df.index[-100:], df['Close'][-100:], label='Historical', color='blue')
                ax_pred.plot(pred_df['Date'], pred_df['Estimated Close'], label='Predicted', color='red')
                ax_pred.fill_between(pred_df['Date'], pred_df['Lower Bound (95% CI)'],
                                     pred_df['Upper Bound (95% CI)'], color='red', alpha=0.1,
                                     label='95% CI')
                ax_pred.set_title(f'{ticker} Prediction ({horizon})')
                ax_pred.legend(); ax_pred.grid(True)
                buf_pred = BytesIO(); fig_pred.savefig(buf_pred, format='png'); buf_pred.seek(0)
                prediction_plot = base64.b64encode(buf_pred.getvalue()).decode()
                plt.close(fig_pred)

    # check if admin mode
    admin = request.args.get("admin", "0") == "1"

    return render_template(
        'dashboard.html',
        error=error,
        historical_plot=historical_plot,
        df_tail=df.tail(10).to_html(classes="table table-sm") if not df.empty else None,
        prediction_plot=prediction_plot,
        pred_table=pred_table,
        overall_pct=overall_pct_str,
        selected_ticker=ticker,
        term=term,
        trading_signal=trading_signal,
        signal_reason=signal_reason,
        signal_confidence=signal_confidence,
        metrics=metrics if admin else None
    )

if __name__ == '__main__':
    app.run(debug=True)
