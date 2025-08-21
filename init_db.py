import yfinance as yf
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler  # Non-blocking for web app integration (e.g., Django)
import pytz
import os
import logging
import time

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("stock_updater.log"),
        logging.StreamHandler()
    ]
)

# 1. Fetch stock data (corrected to fetch per-stock for reliability and correct Stock assignment)
def fetch_stock_data(stocks, start_date=None, end_date=None):
    end_date = end_date or datetime.now().strftime('%Y-%m-%d')
    start_date = start_date or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    stock_data = {}
    for stock in stocks:
        try:
            logging.info(f"Fetching data for stock: {stock} from {start_date} to {end_date}...")
            data = yf.download(stock, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if not data.empty:
                data['Stock'] = stock
                data.index = pd.to_datetime(data.index)
                stock_data[stock] = data
            else:
                logging.warning(f"No data retrieved for {stock}")
        except Exception as e:
            logging.error(f"Error fetching data for {stock}: {e}")
    return stock_data

# 2. Calculate metrics (unchanged, with added check for required columns)
def calculate_metrics(df):
    required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        logging.error(f"Missing required columns in DataFrame: {df.columns}")
        return df  # Return unchanged if columns missing

    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Sharpe_Ratio'] = (df['Daily_Return'].rolling(window=20).mean() * 252) / df['Volatility']
    return df

# 3. Store in SQLite (unchanged)
def store_in_database(stock_data, db_name='D:\\AAPL_STOCK_APP\\stocks.db'):
    for stock, df in stock_data.items():
        # Each stock gets its own database file
        stock_db_name = f"D:/AAPL_STOCK_APP/{stock}.db"
        os.makedirs(os.path.dirname(stock_db_name), exist_ok=True)
        conn = sqlite3.connect(stock_db_name)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_prices (
                Date TEXT,
                Stock TEXT,
                Open REAL,
                High REAL,
                Low REAL,
                Close REAL,
                Adj_Close REAL,
                Volume INTEGER,
                Daily_Return REAL,
                Volatility REAL,
                MA20 REAL,
                MA50 REAL,
                Sharpe_Ratio REAL,
                PRIMARY KEY (Date, Stock)
            )
        ''')
        conn.commit()

        try:
            df = calculate_metrics(df).reset_index()
            df = df.rename(columns={'Adj Close': 'Adj_Close'})
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

            cols = ['Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Adj_Close',
                    'Volume', 'Daily_Return', 'Volatility', 'MA20', 'MA50', 'Sharpe_Ratio']
            rows = df[cols].values.tolist()

            if rows:  # only insert if we have data
                placeholders = ','.join('?' * len(cols))
                query = f"INSERT OR REPLACE INTO stock_prices ({','.join(cols)}) VALUES ({placeholders})"
                cursor.executemany(query, rows)
                logging.info(f"Inserted/updated {len(rows)} rows for {stock} in {stock_db_name}.")
            else:
                logging.warning(f"No rows to insert for {stock}.")
        except Exception as e:
            logging.error(f"Error inserting {stock} into DB {stock_db_name}: {e}")

        conn.commit()
        conn.close()

# 4. Get last stored date (unchanged)
def get_last_date(stock, db_name):
    # Always use the per-stock database
    stock_db_name = f"D:/Users/Nadia/MyProjects/AAPL_STOCK_APP/{stock}.db"
    if not os.path.exists(stock_db_name):
        return None
    conn = sqlite3.connect(stock_db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(Date) FROM stock_prices WHERE Stock = ?", (stock,))
    result = cursor.fetchone()[0]
    conn.close()
    return result

# 5. Daily update: fetch only new rows (unchanged, already per-stock)
def fetch_and_store_daily(stocks, db_name):
    logging.info("Running daily update...")
    stock_data = {}
    for stock in stocks:
        last_date = get_last_date(stock, db_name)
        if last_date:
            start_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            logging.info(f"No existing data for {stock}, fetching full history (5 years)...")
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')

        # Fetch for this stock only
        stock_data.update(fetch_stock_data([stock], start_date=start_date))

    store_in_database(stock_data, db_name)

# 6. Query data (unchanged)
def query_stock_data(stock, db_name='D:\\AAPL_STOCK_APP\\stocks.db'):
    # Always use the per-stock database
    stock_db_name = f"D:/AAPL_STOCK_APP/{stock}.db"
    try:
        conn = sqlite3.connect(stock_db_name)
        df = pd.read_sql("SELECT * FROM stock_prices WHERE Stock = ? ORDER BY Date", conn, params=(stock,))
        conn.close()
        return df
    except Exception as e:
        logging.error(f"Error querying {stock}: {e}")
        return pd.DataFrame()

# 7. Scheduler (non-blocking BackgroundScheduler for web app integration)
def schedule_daily_update(stocks, db_name):
    scheduler = BackgroundScheduler(timezone=pytz.timezone('Europe/Paris'))
    scheduler.add_job(fetch_and_store_daily, 'cron', args=[stocks, db_name], hour=23, minute=0)
    logging.info("Daily scheduler started (non-blocking). Waiting for jobs...")
    scheduler.start()

# 8. Main
if __name__ == "__main__":
    # List your big stocks here
    stocks = ['AAPL']
    db_name = 'D:\\AAPL_STOCK_APP\\stocks.db'

    # First run - populate DB if empty (now uses per-stock fetch for historical too)
    if not os.path.exists(db_name) or query_stock_data(stocks[0], db_name).empty:
        logging.info("Database is empty. Fetching 5 years of historical data...")
        historical_data = fetch_stock_data(
            stocks, 
            start_date=(datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        )
        store_in_database(historical_data, db_name)
    else:
        logging.info("Database already has data. Skipping initial full fetch.")

    # Show sample for each stock
    for stock in stocks:
        df_sample = query_stock_data(stock, db_name)
        if not df_sample.empty:
            logging.info(f"Sample data for {stock}:\n{df_sample.head()}")
        else:
            logging.warning(f"No data found for {stock}.")

    # Start daily scheduler
    schedule_daily_update(stocks, db_name)

    # Keep script alive for scheduler (remove this loop if integrating with Django/web app, as the scheduler runs in background)
    while True:
        time.sleep(60)

# This will keep the script running to allow the scheduler to execute jobs (standalone mode).
# You can stop it with Ctrl+C or by terminating the process.
# For Django integration: Start the scheduler in a management command or app.py (e.g., in ready() method), without the while loop.
# Make sure to run this script in an environment where you can keep it running continuously
# such as a server or a dedicated machine for stock data processing.
# Ensure you have the required packages installed: yfinance, apscheduler, pandas, sqlite3, pytz
# You can install them via pip if needed:
# pip install yfinance apscheduler pandas pytz
# Note: Adjust the database path as needed for your environment.    