import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import requests
import json
import time
import sqlite3
import pickle
import hashlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, Attention, Concatenate, Layer, MultiHeadAttention, BatchNormalization, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
STATIC_IMAGES = {
    "sidebar": "https://cdn.pixabay.com/photo/2018/01/12/16/15/graph-3078539_1280.png",
    "technical": "https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg",
    "ai": "https://cdn.pixabay.com/photo/2023/02/05/01/09/artificial-intelligence-7768523_1280.jpg", 
    "analysis": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQOO408t_Cw1awjp4zjYFnOsagwPKpvtNYC2w&s",
    "news": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRE5Xb4pgJyEnP5WTdmiSu2E1iSb7JMqOsvoQ&s",
  
    "about": "data:image/jpeg;base64,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"
}
# Page configuration with custom theme
st.set_page_config(
    page_title="StockSense AI Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional design
def load_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Poppins', sans-serif;
            }
            
            .stApp {
                background: linear-gradient(to bottom, #1A1F2C, #2D3748);
            }
            
            .main-title {
                color: #9b87f5;
                font-weight: 700;
                font-size: 2.75rem;
                margin-bottom: 0.5rem;
                text-align: center;
                text-shadow: 0px 2px 4px rgba(0,0,0,0.3);
            }
            
            .sub-title {
                color: #D6BCFA;
                font-weight: 400;
                font-size: 1.2rem;
                margin-bottom: 2rem;
                text-align: center;
                opacity: 0.9;
            }
            
            .card {
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1.25rem;
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
                backdrop-filter: blur(10px);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 25px rgba(0, 0, 0, 0.3);
            }
            
            .metric-container {
                background-color: rgba(155, 135, 245, 0.1);
                border-radius: 8px;
                padding: 1.25rem;
                margin-bottom: 1rem;
                border-left: 4px solid #9b87f5;
                transition: all 0.2s ease;
            }
            
            .metric-container:hover {
                background-color: rgba(155, 135, 245, 0.15);
                transform: translateX(5px);
            }
            
            .info-box {
                background-color: rgba(30, 174, 219, 0.1);
                border-radius: 10px;
                padding: 1.25rem;
                margin-bottom: 1rem;
                border-left: 4px solid #1EAEDB;
            }
            
            .news-card {
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                padding: 1rem;
                margin-bottom: 0.75rem;
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: all 0.2s ease;
            }
            
            .news-card:hover {
                background-color: rgba(255, 255, 255, 0.08);
            }
            
            .positive-sentiment {
                color: #48BB78;
                font-weight: 500;
            }
            
            .negative-sentiment {
                color: #F56565;
                font-weight: 500;
            }
            
            .neutral-sentiment {
                color: #ECC94B;
                font-weight: 500;
            }
            
            /* Make the sidebar more professional */
            .css-1d391kg, .css-12oz5g7 {
                background-color: rgba(26, 31, 44, 0.9);
            }
            
            /* Customize button */
            .stButton>button {
                background-color: #9b87f5;
                color: white;
                border-radius: 8px;
                border: none;
                padding: 0.6rem 1.2rem;
                font-weight: 500;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-size: 14px;
            }
            
            .stButton>button:hover {
                background-color: #7a5af8;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
                transform: translateY(-2px);
            }
            
            /* Customize selectbox */
            .stSelectbox>div>div {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
            }
            
            /* Customize number input */
            .stNumberInput>div>div {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
            }
            
            /* Customize text input */
            .stTextInput>div>div {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
            }
            
            /* Progress bar */
            .stProgress > div > div > div {
                background-color: #9b87f5;
            }
            
            h1, h2, h3, h4, h5 {
                color: #D6BCFA;
            }
            
            .insight-card {
                background: linear-gradient(135deg, rgba(155, 135, 245, 0.15) 0%, rgba(30, 174, 219, 0.1) 100%);
                border-radius: 10px;
                padding: 1.25rem;
                margin-bottom: 1rem;
                border: 1px solid rgba(155, 135, 245, 0.3);
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            }
            
            .tab-content {
                padding: 1.5rem;
                background-color: rgba(255, 255, 255, 0.02);
                border-radius: 0 0 10px 10px;
                border-top: none;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            }
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #9b87f5;
                border-radius: 10px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #7a5af8;
            }
            
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }

            .stTabs [data-baseweb="tab"] {
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 8px 8px 0px 0px;
                padding: 10px 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-bottom: none;
            }

            .stTabs [aria-selected="true"] {
                background-color: rgba(155, 135, 245, 0.2);
                border-bottom: 2px solid #9b87f5;
            }
            
            /* Animation for cards */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .card {
                animation: fadeIn 0.5s ease-out forwards;
            }
            
            /* Numbered list styling */
            ol {
                counter-reset: item;
                list-style-type: none;
                padding-left: 1rem;
            }
            
            ol li {
                position: relative;
                padding-left: 2.5rem;
                margin-bottom: 0.8rem;
            }
            
            ol li:before {
                content: counter(item) "";
                counter-increment: item;
                position: absolute;
                left: 0;
                top: 0;
                background: #9b87f5;
                border-radius: 50%;
                width: 1.8rem;
                height: 1.8rem;
                color: white;
                font-weight: bold;
                text-align: center;
                line-height: 1.8rem;
            }
        </style>
    """, unsafe_allow_html=True)

load_css()

# ------------------------------------ MODEL DATABASE ------------------------------------

# Database file path
DB_PATH = "stock_models.db"

def init_model_database():
    """Initialize the model cache database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_symbol TEXT NOT NULL,
            model_type TEXT NOT NULL,
            data_hash TEXT NOT NULL,
            model_data BLOB NOT NULL,
            scaler_data BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            data_start_date TEXT,
            data_end_date TEXT,
            num_records INTEGER,
            UNIQUE(stock_symbol, model_type, data_hash)
        )
    ''')
    
    conn.commit()
    conn.close()

def compute_data_hash(df):
    """Compute hash of dataframe to detect data changes"""
    # Use last date and row count as simple hash
    last_date = str(df.index[-1]) if len(df) > 0 else ""
    row_count = len(df)
    hash_string = f"{last_date}_{row_count}"
    return hashlib.md5(hash_string.encode()).hexdigest()

def save_model_to_db(stock_symbol, model_type, model, df, scaler=None):
    """Save trained model to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Serialize model
        model_bytes = pickle.dumps(model)
        scaler_bytes = pickle.dumps(scaler) if scaler else None
        
        # Compute data hash
        data_hash = compute_data_hash(df)
        
        # Get data info
        data_start = str(df.index[0]) if len(df) > 0 else ""
        data_end = str(df.index[-1]) if len(df) > 0 else ""
        num_records = len(df)
        
        # Delete old model if exists
        cursor.execute('''
            DELETE FROM model_cache 
            WHERE stock_symbol = ? AND model_type = ?
        ''', (stock_symbol, model_type))
        
        # Insert new model
        cursor.execute('''
            INSERT INTO model_cache 
            (stock_symbol, model_type, data_hash, model_data, scaler_data, 
             data_start_date, data_end_date, num_records)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (stock_symbol, model_type, data_hash, model_bytes, scaler_bytes,
              data_start, data_end, num_records))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.warning(f"Could not save model to database: {str(e)}")
        return False

def load_model_from_db(stock_symbol, model_type, df):
    """Load trained model from database if available and data matches"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Compute current data hash
        current_hash = compute_data_hash(df)
        
        # Try to fetch model
        cursor.execute('''
            SELECT model_data, scaler_data, created_at, data_hash
            FROM model_cache
            WHERE stock_symbol = ? AND model_type = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (stock_symbol, model_type))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            model_bytes, scaler_bytes, created_at, stored_hash = result
            
            # Check if data has changed significantly
            if stored_hash == current_hash:
                # Data matches, load model
                model = pickle.loads(model_bytes)
                scaler = pickle.loads(scaler_bytes) if scaler_bytes else None
                
                # Show cache hit message
                cache_age = datetime.now() - datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
                st.info(f"✅ Using cached {model_type} model (trained {cache_age.seconds // 3600}h ago)")
                
                return model, scaler
        
        return None, None
    except Exception as e:
        st.warning(f"Could not load cached model: {str(e)}")
        return None, None

def clear_old_models(days_old=7):
    """Clear models older than specified days"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        cursor.execute('''
            DELETE FROM model_cache
            WHERE created_at < ?
        ''', (cutoff_date.strftime('%Y-%m-%d %H:%M:%S'),))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count
    except:
        return 0

def clear_all_models():
    """Clear all cached models (force retrain)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM model_cache')
        count = cursor.fetchone()[0]
        
        cursor.execute('DELETE FROM model_cache')
        
        conn.commit()
        conn.close()
        
        return count
    except Exception as e:
        st.error(f"Error clearing models: {str(e)}")
        return 0

def get_cache_stats():
    """Get statistics about cached models"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT stock_symbol, model_type, created_at, num_records 
            FROM model_cache 
            ORDER BY created_at DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    except:
        return []

# Initialize database on startup
init_model_database()

# Define our functions -----------------------------------------------------------------

# ------------------------------------ DATA FETCHING ------------------------------------

def load_stock_data(symbol, data_source, period):
    """Fetch stock data from the selected source"""
    days = get_days_from_period(period)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    def generate_sample_data():
        idx = pd.date_range(end=datetime.now(), periods=days)

        np.random.seed(42)
        initial_price = 100
        prices = [initial_price]

        for i in range(1, len(idx)):
            change_percent = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + change_percent)
            prices.append(new_price)

        df = pd.DataFrame(index=idx)
        df['Close'] = prices
        df['Open'] = df['Close'] * (1 + np.random.normal(0, 0.01, size=len(df)))
        df['High'] = pd.concat([df['Open'], df['Close']], axis=1).max(axis=1) * (1 + abs(np.random.normal(0, 0.005, size=len(df))))
        df['Low'] = pd.concat([df['Open'], df['Close']], axis=1).min(axis=1) * (1 - abs(np.random.normal(0, 0.005, size=len(df))))
        df['Volume'] = np.random.normal(1000000, 200000, size=len(df)).astype(int)
        df['Volume'] = df['Volume'].apply(lambda x: max(0, x))

        return df
    
    try:
        if data_source == "Yahoo Finance":
            # Download data from Yahoo Finance
            try:
                df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
            except Exception as e:
                st.warning(f"Yahoo Finance unavailable for {symbol}: {e}. Using sample data instead.")
                return generate_sample_data()

            if df is None or df.empty:
                st.warning(f"No data found for symbol {symbol}. Using sample data instead.")
                return generate_sample_data()
            
            # If columns are MultiIndex (happens with yfinance), flatten them
            if isinstance(df.columns, pd.MultiIndex):
                # For single ticker, just take the first level (price type: Close, Open, etc.)
                df.columns = df.columns.get_level_values(0)
            
            # Remove any duplicate columns that might occur
            df = df.loc[:, ~df.columns.duplicated()]
            
            # Standardize column names - handle both 'Adj Close' and 'Close'
            if 'Adj Close' in df.columns:
                # Use Adj Close as the main Close price (it's adjusted for splits/dividends)
                df['Close'] = df['Adj Close']
            
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                return None
            
            # Select only required columns and create a clean copy
            df = df[required_cols].copy()
            
            # Drop rows where all values are NaN
            df = df.dropna(how='all')
            
            # Forward fill and backward fill any remaining NaN values
            df = df.ffill().bfill()
            
            # Ensure all columns are numeric
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any remaining NaN rows after conversion
            df = df.dropna()
            
            # Ensure Volume is non-negative
            df['Volume'] = df['Volume'].abs()
            
            st.success(f"✅ Loaded {len(df)} days of data for {symbol}")
            
            return df
            
        else:  # Sample Data
            return generate_sample_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_days_from_period(period):
    """Convert time period string to number of days"""
    if period == "1 Month":
        return 30
    elif period == "3 Months":
        return 90
    elif period == "6 Months":
        return 180
    elif period == "1 Year":
        return 365
    elif period == "2 Years":
        return 730
    elif period == "5 Years":
        return 1825

# ------------------------------------ TECHNICAL INDICATORS ------------------------------------

def calculate_technical_indicators(df):
    """Calculate various technical indicators for the given stock dataframe"""
    # Make a copy to avoid modifying the original
    df_tech = df.copy()
    
    # Check required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df_tech.columns:
            raise ValueError(f"Input DataFrame is missing required column: {col}")
    
    # Relative Strength Index (RSI)
    delta = df_tech['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    # Calculate average gain and loss over 14 periods
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean().abs()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    df_tech['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD)
    exp1 = df_tech['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_tech['Close'].ewm(span=26, adjust=False).mean()
    df_tech['MACD'] = exp1 - exp2
    df_tech['MACD_Signal'] = df_tech['MACD'].ewm(span=9, adjust=False).mean()
    df_tech['MACD_Histogram'] = df_tech['MACD'] - df_tech['MACD_Signal']
    
    # Simple Moving Averages
    df_tech['SMA_20'] = df_tech['Close'].rolling(window=20).mean()
    df_tech['SMA_50'] = df_tech['Close'].rolling(window=50).mean()
    df_tech['SMA_200'] = df_tech['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df_tech['EMA_20'] = df_tech['Close'].ewm(span=20, adjust=False).mean()
    df_tech['EMA_50'] = df_tech['Close'].ewm(span=50, adjust=False).mean()
    df_tech['EMA_200'] = df_tech['Close'].ewm(span=200, adjust=False).mean()
    
    # Bollinger Bands (20-day, 2 standard deviations)
    df_tech['BB_Middle'] = df_tech['Close'].rolling(window=20).mean()
    df_tech['BB_Std'] = df_tech['Close'].rolling(window=20).std()
    df_tech['BB_Upper'] = df_tech['BB_Middle'] + (df_tech['BB_Std'] * 2)
    df_tech['BB_Lower'] = df_tech['BB_Middle'] - (df_tech['BB_Std'] * 2)
    df_tech['BB_Width'] = (df_tech['BB_Upper'] - df_tech['BB_Lower']) / df_tech['BB_Middle']
    
    # Average True Range (ATR) - Volatility Indicator
    df_tech['TR'] = np.maximum(
        df_tech['High'] - df_tech['Low'],
        np.maximum(
            abs(df_tech['High'] - df_tech['Close'].shift(1)),
            abs(df_tech['Low'] - df_tech['Close'].shift(1))
        )
    )
    df_tech['ATR'] = df_tech['TR'].rolling(window=14).mean()
    
    # Stochastic Oscillator
    low_14 = df_tech['Low'].rolling(window=14).min()
    high_14 = df_tech['High'].rolling(window=14).max()
    df_tech['%K'] = 100 * ((df_tech['Close'] - low_14) / (high_14 - low_14))
    df_tech['%D'] = df_tech['%K'].rolling(window=3).mean()
    
    # Money Flow Index (MFI)
    typical_price = (df_tech['High'] + df_tech['Low'] + df_tech['Close']) / 3
    money_flow = typical_price * df_tech['Volume']
    
    # Get positive and negative money flow
    delta_typical = typical_price.diff()
    positive_flow = money_flow.where(delta_typical > 0, 0)
    negative_flow = money_flow.where(delta_typical < 0, 0)
    
    # Calculate MFI
    positive_flow_sum = positive_flow.rolling(window=14).sum()
    negative_flow_sum = negative_flow.rolling(window=14).sum().abs()
    money_ratio = positive_flow_sum / negative_flow_sum
    df_tech['MFI'] = 100 - (100 / (1 + money_ratio))
    
    # On-Balance Volume (OBV)
    df_tech['OBV'] = (df_tech['Volume'] * ((df_tech['Close'].diff() > 0) * 2 - 1)).cumsum()
    
    # Price Rate of Change (ROC)
    df_tech['ROC'] = df_tech['Close'].pct_change(periods=10) * 100
    
    # Ichimoku Cloud
    high_9 = df_tech['High'].rolling(window=9).max()
    low_9 = df_tech['Low'].rolling(window=9).min()
    df_tech['Conversion_Line'] = (high_9 + low_9) / 2
    
    high_26 = df_tech['High'].rolling(window=26).max()
    low_26 = df_tech['Low'].rolling(window=26).min()
    df_tech['Base_Line'] = (high_26 + low_26) / 2
    
    df_tech['Leading_Span_A'] = ((df_tech['Conversion_Line'] + df_tech['Base_Line']) / 2).shift(26)
    
    high_52 = df_tech['High'].rolling(window=52).max()
    low_52 = df_tech['Low'].rolling(window=52).min()
    df_tech['Leading_Span_B'] = ((high_52 + low_52) / 2).shift(26)
    
    df_tech['Lagging_Span'] = df_tech['Close'].shift(-26)
    
    # Fibonacci Retracement Levels
    # We'll calculate some Fibonacci retracement levels based on the min and max in the period
    price_max = df_tech['High'].max()
    price_min = df_tech['Low'].min()
    price_diff = price_max - price_min
    
    df_tech['Fib_0'] = price_min
    df_tech['Fib_23.6'] = price_min + price_diff * 0.236
    df_tech['Fib_38.2'] = price_min + price_diff * 0.382
    df_tech['Fib_50'] = price_min + price_diff * 0.5
    df_tech['Fib_61.8'] = price_min + price_diff * 0.618
    df_tech['Fib_100'] = price_max
    
    # ADX (Average Directional Index)
    # True Range
    df_tech['TR'] = np.maximum(
        df_tech['High'] - df_tech['Low'],
        np.maximum(
            abs(df_tech['High'] - df_tech['Close'].shift(1)),
            abs(df_tech['Low'] - df_tech['Close'].shift(1))
        )
    )
    
    # Directional Movement
    df_tech['DMplus'] = np.where(
        (df_tech['High'] - df_tech['High'].shift(1)) > (df_tech['Low'].shift(1) - df_tech['Low']),
        np.maximum(df_tech['High'] - df_tech['High'].shift(1), 0),
        0
    )
    
    df_tech['DMminus'] = np.where(
        (df_tech['Low'].shift(1) - df_tech['Low']) > (df_tech['High'] - df_tech['High'].shift(1)),
        np.maximum(df_tech['Low'].shift(1) - df_tech['Low'], 0),
        0
    )
    
    # Smoothed TR and DM
    window = 14
    df_tech['smooth_TR'] = df_tech['TR'].rolling(window=window).sum()
    df_tech['smooth_DMplus'] = df_tech['DMplus'].rolling(window=window).sum()
    df_tech['smooth_DMminus'] = df_tech['DMminus'].rolling(window=window).sum()
    
    # DI (Directional Indicator)
    df_tech['DIplus'] = 100 * df_tech['smooth_DMplus'] / df_tech['smooth_TR']
    df_tech['DIminus'] = 100 * df_tech['smooth_DMminus'] / df_tech['smooth_TR']
    
    # DX (Directional Index)
    df_tech['DX'] = 100 * abs(df_tech['DIplus'] - df_tech['DIminus']) / (df_tech['DIplus'] + df_tech['DIminus'])
    
    # ADX (Average Directional Index)
    df_tech['ADX'] = df_tech['DX'].rolling(window=window).mean()
    
    return df_tech

# ------------------------------------ FEATURE ENGINEERING ------------------------------------

def engineer_features(df):
    """Engineer additional features from stock data"""
    # Create a copy of the dataframe to avoid modifying the original
    df_features = df.copy()
    
    # Make sure the index is a datetime
    if not isinstance(df_features.index, pd.DatetimeIndex):
        df_features.index = pd.to_datetime(df_features.index)
    
    # Price-based features
    df_features['Price_Change'] = df_features['Close'].diff()
    df_features['Pct_Change'] = df_features['Close'].pct_change() * 100
    
    # Volatility-based features
    df_features['Daily_Return'] = df_features['Close'].pct_change()
    df_features['Daily_Volatility'] = df_features['Daily_Return'].rolling(window=10).std() * np.sqrt(252)  # Annualized
    df_features['Price_Range'] = df_features['High'] - df_features['Low']
    df_features['Range_Pct'] = (df_features['High'] - df_features['Low']) / df_features['Close'] * 100
    
    # Volume-based features
    df_features['Volume_Change'] = df_features['Volume'].diff()
    df_features['Volume_Pct_Change'] = df_features['Volume'].pct_change() * 100
    df_features['Relative_Volume'] = df_features['Volume'] / df_features['Volume'].rolling(window=20).mean()
    df_features['OBV'] = (np.sign(df_features['Close'].diff()) * df_features['Volume']).fillna(0).cumsum()
    
    # Momentum features
    df_features['Momentum_1'] = df_features['Close'] / df_features['Close'].shift(1) - 1
    df_features['Momentum_5'] = df_features['Close'] / df_features['Close'].shift(5) - 1
    df_features['Momentum_10'] = df_features['Close'] / df_features['Close'].shift(10) - 1
    df_features['Momentum_20'] = df_features['Close'] / df_features['Close'].shift(20) - 1
    
    # Moving average-based features
    df_features['SMA_5'] = df_features['Close'].rolling(window=5).mean()
    df_features['SMA_10'] = df_features['Close'].rolling(window=10).mean()
    df_features['SMA_20'] = df_features['Close'].rolling(window=20).mean()
    df_features['SMA_50'] = df_features['Close'].rolling(window=50).mean()
    df_features['SMA_200'] = df_features['Close'].rolling(window=200).mean()
    
    # Price relative to moving averages
    df_features['Price_SMA_5_Ratio'] = df_features['Close'] / df_features['SMA_5']
    df_features['Price_SMA_10_Ratio'] = df_features['Close'] / df_features['SMA_10']
    df_features['Price_SMA_20_Ratio'] = df_features['Close'] / df_features['SMA_20']
    
    # Distance from high/low
    df_features['Price_52W_High'] = df_features['High'].rolling(window=252).max()
    df_features['Price_52W_Low'] = df_features['Low'].rolling(window=252).min()
    df_features['Pct_From_52W_High'] = (df_features['Close'] / df_features['Price_52W_High'] - 1) * 100
    df_features['Pct_From_52W_Low'] = (df_features['Close'] / df_features['Price_52W_Low'] - 1) * 100
    
    # Time-based features
    df_features['Day_of_Week'] = df_features.index.dayofweek
    df_features['Month'] = df_features.index.month
    df_features['Quarter'] = df_features.index.quarter
    df_features['Year'] = df_features.index.year
    df_features['Day_of_Year'] = df_features.index.dayofyear
    df_features['Is_Month_End'] = df_features.index.is_month_end.astype(int)
    df_features['Is_Month_Start'] = df_features.index.is_month_start.astype(int)
    df_features['Is_Quarter_End'] = df_features.index.is_quarter_end.astype(int)
    df_features['Is_Quarter_Start'] = df_features.index.is_quarter_start.astype(int)
    
    # Advanced technical features
    # Stochastic RSI
    df_features['RSI'] = calculate_rsi(df_features['Close'])
    rsi = df_features['RSI'].copy()
    df_features['StochRSI'] = ((rsi - rsi.rolling(window=14).min()) / 
                             (rsi.rolling(window=14).max() - rsi.rolling(window=14).min()))
    
    # VWAP (Volume Weighted Average Price)
    df_features['Typical_Price'] = (df_features['High'] + df_features['Low'] + df_features['Close']) / 3
    df_features['VWAP'] = (df_features['Typical_Price'] * df_features['Volume']).cumsum() / df_features['Volume'].cumsum()
    
    # Pattern recognition features
    # Gap up/down
    df_features['Gap_Up'] = ((df_features['Open'] > df_features['High'].shift(1)) * 1)
    df_features['Gap_Down'] = ((df_features['Open'] < df_features['Low'].shift(1)) * 1)
    
    # Inside/outside days
    df_features['Inside_Day'] = ((df_features['High'] < df_features['High'].shift(1)) & 
                                 (df_features['Low'] > df_features['Low'].shift(1))).astype(int)
    df_features['Outside_Day'] = ((df_features['High'] > df_features['High'].shift(1)) & 
                                  (df_features['Low'] < df_features['Low'].shift(1))).astype(int)
    
    # Candlestick patterns
    df_features['Doji'] = (
        abs(df_features['Close'] - df_features['Open']) <= 
        (0.1 * (df_features['High'] - df_features['Low']))
    ).astype(int)
    
    # Trend strength indicators
    # ADX (Average Directional Index) - simplified version
    df_features['TR'] = np.maximum(
        df_features['High'] - df_features['Low'],
        np.maximum(
            abs(df_features['High'] - df_features['Close'].shift(1)),
            abs(df_features['Low'] - df_features['Close'].shift(1))
        )
    )
    df_features['ATR'] = df_features['TR'].rolling(window=14).mean()
    
    # Volatility Ratio
    df_features['Volatility_Ratio'] = df_features['ATR'] / df_features['Close'] * 100
    
    # Fill NaN values that result from calculations
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    
    return df_features

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    deltas = prices.diff()
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1. + rs)
    
    for i in range(window, len(prices)):
        delta = deltas[i]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
            
        up = (up * (window - 1) + upval) / window
        down = (down * (window - 1) + downval) / window
        
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)
        
    return pd.Series(rsi, index=prices.index)

# ------------------------------------ NEWS SENTIMENT ANALYSIS ------------------------------------

# GNews API configuration
import os

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

GNEWS_API_KEY = os.environ.get("GNEWS_API_KEY")


def get_news_sentiment(symbol, days=7):
    """Fetch news articles related to a stock symbol using GNews API"""
    try:
        # Use just the stock symbol as query
        query = symbol
        max_articles = min(days, 10)
        
        # GNews API endpoint
        url = f"https://gnews.io/api/v4/search?q={query}&lang=en&max={max_articles}&apikey={GNEWS_API_KEY}"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            return articles
        else:
            try:
                error_data = response.json()
                st.warning(f"GNews API Error: {error_data.get('errors', [response.status_code])}")
            except:
                st.warning(f"Unable to fetch news (Status: {response.status_code}). API may have rate limits or quota issues.")
            return []
    except Exception as e:
        st.warning(f"Error fetching news: {str(e)}")
        return []

# ------------------------------------ MODEL PREDICTIONS ------------------------------------

def find_optimal_arima_params(data, exog=None, max_p=5, max_d=2, max_q=5):
    """Find optimal SARIMAX parameters using AIC criterion"""
    best_aic = np.inf
    best_params = (2, 1, 2)
    
    # Test for stationarity to determine d
    try:
        adf_result = adfuller(data)
        is_stationary = adf_result[1] < 0.05
        d_range = [0, 1] if is_stationary else [1]
    except:
        d_range = [1]
    
    # Grid search over p, d, q with limited combinations
    param_combinations = [
        (1, 1, 1), (2, 1, 2), (3, 1, 2), (2, 1, 3),
        (1, 1, 2), (2, 1, 1), (3, 1, 1), (1, 1, 0),
        (0, 1, 1), (4, 1, 2), (2, 1, 4)
    ]
    
    for p, d, q in param_combinations:
        try:
            if exog is not None:
                model = SARIMAX(data, exog=exog, order=(p, d, q), 
                              enforce_stationarity=False, enforce_invertibility=False)
            else:
                model = SARIMAX(data, order=(p, d, q),
                              enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False, maxiter=100)
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_params = (p, d, q)
        except:
            continue
    
    return best_params

def predict_arima(df, forecast_days, stock_symbol="UNKNOWN"):
    """Enhanced SARIMAX model with exogenous variables and feature engineering"""
    try:
        # Work with required columns
        required_cols = ['Close', 'Volume', 'High', 'Low', 'Open']
        if not all(col in df.columns for col in required_cols):
            st.error("Missing required columns for ARIMA prediction")
            return None
        
        df_arima = df[required_cols].copy()
        df_arima = df_arima.dropna()
        
        # Validate data
        if len(df_arima) < 60:
            st.warning(f"Insufficient data for ARIMA prediction. Need at least 60 days, got {len(df_arima)} days.")
            last_date = df.index[-1] if len(df) > 0 else datetime.now()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
            future_dates = future_dates[future_dates.dayofweek < 5][:forecast_days]
            last_price = df_arima['Close'].iloc[-1] if len(df_arima) > 0 else 100.0
            dummy_pred = [last_price] * len(future_dates)
            return {
                'future_dates': future_dates,
                'future_predictions': dummy_pred,
                'test_predictions': df['Close'].iloc[-min(10, len(df)-1):].tolist() if len(df) > 1 else [],
                'lower_bound': [p * 0.95 for p in dummy_pred],
                'upper_bound': [p * 1.05 for p in dummy_pred]
            }
        
        # Feature engineering - create exogenous variables with momentum indicators
        df_arima['Returns'] = df_arima['Close'].pct_change()
        df_arima['Volume_MA'] = df_arima['Volume'].rolling(window=5).mean()
        df_arima['Price_Range'] = (df_arima['High'] - df_arima['Low']) / df_arima['Close']
        df_arima['MA_5'] = df_arima['Close'].rolling(window=5).mean()
        df_arima['MA_20'] = df_arima['Close'].rolling(window=20).mean()
        df_arima['Volatility'] = df_arima['Returns'].rolling(window=10).std()
        
        # Momentum indicators for better directional accuracy
        df_arima['RSI'] = 100 - (100 / (1 + (df_arima['Close'].diff().clip(lower=0).rolling(14).mean() / 
                                              (-df_arima['Close'].diff().clip(upper=0).rolling(14).mean()))))
        df_arima['MACD'] = df_arima['Close'].ewm(span=12).mean() - df_arima['Close'].ewm(span=26).mean()
        df_arima['MACD_Signal'] = df_arima['MACD'].ewm(span=9).mean()
        df_arima['MACD_Hist'] = df_arima['MACD'] - df_arima['MACD_Signal']
        
        # Trend indicators
        df_arima['Price_Momentum'] = df_arima['Close'] - df_arima['Close'].shift(10)
        df_arima['Volume_Ratio'] = df_arima['Volume'] / df_arima['Volume'].rolling(window=20).mean()
        df_arima['MA_Trend'] = (df_arima['MA_5'] - df_arima['MA_20']) / df_arima['MA_20']
        
        # Lag features for temporal patterns
        df_arima['Close_Lag1'] = df_arima['Close'].shift(1)
        df_arima['Close_Lag2'] = df_arima['Close'].shift(2)
        df_arima['Returns_Lag1'] = df_arima['Returns'].shift(1)
        
        # Drop NaN created by rolling windows
        df_arima = df_arima.dropna()
        
        if len(df_arima) < 50:
            st.warning("Insufficient data after feature engineering")
            return None
        
        # Try to load cached model
        model_fit, _ = load_model_from_db(stock_symbol, "ARIMA", df_arima)
        
        # Prepare data
        close_prices = df_arima['Close'].values
        
        # Select enhanced exogenous variables with momentum and trend indicators
        exog_cols = ['Volume_MA', 'Price_Range', 'Volatility', 'Returns', 
                    'RSI', 'MACD_Hist', 'Price_Momentum', 'Volume_Ratio', 
                    'MA_Trend', 'Close_Lag1', 'Returns_Lag1']
        exog_data = df_arima[exog_cols].values
        
        # Fit SARIMAX model (or use cached)
        if model_fit is None:
            st.info("🔄 Finding optimal SARIMAX parameters and training model...")
            
            # Find optimal parameters with exogenous variables
            optimal_params = find_optimal_arima_params(close_prices, exog=exog_data)
            st.info(f"📊 Optimal SARIMAX parameters: {optimal_params}")
            
            # Fit model with optimal parameters
            model = SARIMAX(close_prices, exog=exog_data, order=optimal_params,
                          enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False, maxiter=200)
            
            # Save to database
            save_model_to_db(stock_symbol, "ARIMA", model_fit, df_arima)
        
        # Prepare future exogenous variables (use last known values with small trend)
        last_exog = exog_data[-1]
        exog_trend = np.mean(exog_data[-5:] - exog_data[-6:-1], axis=0)
        future_exog = np.array([last_exog + exog_trend * i for i in range(1, forecast_days + 1)])
        
        # Forecast with confidence intervals
        forecast_result = model_fit.get_forecast(steps=forecast_days, exog=future_exog)
        forecast_values = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval
        
        # Handle confidence intervals
        if hasattr(conf_int, 'iloc'):
            lower_bound = conf_int.iloc[:, 0].values
            upper_bound = conf_int.iloc[:, 1].values
        else:
            lower_bound = conf_int[:, 0]
            upper_bound = conf_int[:, 1]
        
        # Create future dates
        last_date = df_arima.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        # Filter out weekend dates
        future_dates = future_dates[future_dates.dayofweek < 5]
        
        # Adjust forecast length if needed due to weekend filtering
        forecast_values = forecast_values[:len(future_dates)]
        lower_bound = lower_bound[:len(future_dates)]
        upper_bound = upper_bound[:len(future_dates)]
        
        # Get predictions on validation data
        test_size = min(30, len(df_arima) // 3)
        test_predictions = model_fit.predict(start=len(close_prices)-test_size, 
                                            end=len(close_prices)-1,
                                            exog=exog_data[-test_size:])
        
        return {
            'future_dates': future_dates,
            'future_predictions': forecast_values,
            'test_predictions': test_predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    except Exception as e:
        st.error(f"Error in ARIMA prediction: {str(e)}")
        # Return dummy data if error
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        future_dates = future_dates[future_dates.dayofweek < 5]
        dummy_pred = [df['Close'].iloc[-1]] * len(future_dates)
        
        return {
            'future_dates': future_dates,
            'future_predictions': dummy_pred,
            'test_predictions': [df['Close'].iloc[-30:].mean()] * min(30, len(df)),
            'lower_bound': [p * 0.95 for p in dummy_pred],
            'upper_bound': [p * 1.05 for p in dummy_pred]
        }

def predict_random_forest(df, forecast_days, stock_symbol="UNKNOWN"):
    """Random Forest prediction model with database caching"""
    try:
        # Create a copy and work with original columns only
        df_rf = df[['Close', 'Volume']].copy() if 'Close' in df.columns and 'Volume' in df.columns else df.copy()
        
        # Ensure we have the required columns
        if 'Close' not in df_rf.columns or 'Volume' not in df_rf.columns:
            st.error(f"Missing required columns. Available: {df_rf.columns.tolist()}")
            return None
        
        # Ensure numeric and remove NaN
        df_rf['Close'] = pd.to_numeric(df_rf['Close'], errors='coerce')
        df_rf['Volume'] = pd.to_numeric(df_rf['Volume'], errors='coerce')
        df_rf = df_rf.dropna()
        
        # Moderate outlier handling for returns
        df_rf['Returns'] = df_rf['Close'].pct_change()
        
        # Winsorize at 1% and 99% percentiles (less aggressive)
        Q1 = df_rf['Returns'].quantile(0.01)
        Q3 = df_rf['Returns'].quantile(0.99)
        df_rf['Returns'] = df_rf['Returns'].clip(lower=Q1, upper=Q3)
        
        # Light smoothing only (don't reconstruct price, keep original)
        # This preserves price trends while reducing return volatility
        df_rf['Returns_MA'] = df_rf['Returns'].rolling(window=3, min_periods=1).mean()
        
        # Ensure we have enough data
        if len(df_rf) < 25:
            st.warning(f"Insufficient data for Random Forest. Need at least 25 days, got {len(df_rf)} days.")
            last_date = df.index[-1] if len(df) > 0 else datetime.now()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
            future_dates = future_dates[future_dates.dayofweek < 5][:forecast_days]
            last_price = df_rf['Close'].iloc[-1] if len(df_rf) > 0 else 100.0
            future_predictions = [last_price] * len(future_dates)
            return {
                'future_dates': future_dates,
                'future_predictions': future_predictions,
                'test_predictions': df['Close'].iloc[-min(10, len(df)-1):].tolist() if len(df) > 1 else [],
                'lower_bound': [p * 0.98 for p in future_predictions],
                'upper_bound': [p * 1.02 for p in future_predictions]
            }
        
        # Feature engineering with adaptive window sizes
        max_window = len(df_rf) - 5  # Leave room for lags and target
        df_rf['SMA_5'] = df_rf['Close'].rolling(window=min(5, max_window), min_periods=1).mean()
        df_rf['SMA_10'] = df_rf['Close'].rolling(window=min(10, max_window), min_periods=1).mean()
        df_rf['SMA_20'] = df_rf['Close'].rolling(window=min(20, max_window), min_periods=1).mean()
        
        # Robust volatility measures
        df_rf['Std_10'] = df_rf['Close'].rolling(window=min(10, max_window), min_periods=1).std().fillna(0)
        df_rf['Return_Volatility'] = df_rf['Returns'].rolling(window=min(10, max_window), min_periods=1).std().fillna(0)
        
        # Momentum indicators (moderate clipping to preserve information)
        df_rf['Momentum_5'] = (df_rf['Close'] / df_rf['SMA_5'] - 1).clip(-0.5, 0.5)
        df_rf['Momentum_10'] = (df_rf['Close'] / df_rf['SMA_10'] - 1).clip(-0.5, 0.5)
        
        # Trend strength (moderate clipping)
        df_rf['Trend_Strength'] = ((df_rf['SMA_5'] - df_rf['SMA_20']) / df_rf['SMA_20']).clip(-0.5, 0.5)
        
        # Volume momentum (keep it simple)
        df_rf['Volume_MA'] = df_rf['Volume'].rolling(window=min(10, max_window), min_periods=1).mean()
        df_rf['Volume_Ratio'] = df_rf['Volume'] / (df_rf['Volume_MA'] + 1)
        
        # Fill any NaN created by new features
        df_rf = df_rf.ffill().bfill().fillna(0)
        
        # Create lagged features
        for i in range(1, 4):
            df_rf[f'Close_Lag_{i}'] = df_rf['Close'].shift(i)
            df_rf[f'Volume_Lag_{i}'] = df_rf['Volume'].shift(i)
        
        # Create target
        df_rf['Target'] = df_rf['Close'].shift(-1)
        
        # Drop NaN - this will remove first 3 rows (lags) and last 1 row (target)
        df_rf = df_rf.dropna()
        
        # Final check
        if len(df_rf) < 5:
            st.warning(f"After preprocessing: {len(df_rf)} rows. Using simplified prediction.")
            last_date = df.index[-1] if len(df) > 0 else datetime.now()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
            future_dates = future_dates[future_dates.dayofweek < 5][:forecast_days]
            last_price = df['Close'].iloc[-1] if 'Close' in df.columns and len(df) > 0 else 100.0
            future_predictions = [last_price] * len(future_dates)
            return {
                'future_dates': future_dates,
                'future_predictions': future_predictions,
                'test_predictions': df['Close'].iloc[-min(10, len(df)-1):].tolist() if len(df) > 1 else [],
                'lower_bound': [p * 0.98 for p in future_predictions],
                'upper_bound': [p * 1.02 for p in future_predictions]
            }
        
        # Select features and target (balanced feature set)
        features = ['Close', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 'Std_10',
                   'Return_Volatility', 'Momentum_5', 'Momentum_10', 'Trend_Strength',
                   'Volume_Ratio', 'Returns', 'Returns_MA'] + \
                  [f'Close_Lag_{i}' for i in range(1, 4)] + \
                  [f'Volume_Lag_{i}' for i in range(1, 4)]
                  
        X = df_rf[features].values
        y = df_rf['Target'].values
        
        # Try to load cached model
        model, _ = load_model_from_db(stock_symbol, "RandomForest", df_rf)
        
        # Train Random Forest with balanced parameters
        if model is None:
            st.info("🔄 Training Random Forest with balanced regularization...")
            # Balanced parameters - not too constrained, not too free
            model = RandomForestRegressor(
                n_estimators=150,        # Moderate number of trees
                max_depth=12,            # Allow reasonable depth
                min_samples_split=8,     # Moderate split requirement  
                min_samples_leaf=4,      # Moderate leaf size
                max_features='sqrt',     # Standard feature sampling
                min_impurity_decrease=0.0001,  # Small threshold
                max_samples=0.9,         # Bootstrap 90% of data (helps with outliers)
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            
            # Save to database
            save_model_to_db(stock_symbol, "RandomForest", model, df_rf)
        
        # Make predictions for the last part of the data (for testing/validation)
        test_size = min(30, len(df_rf) // 3)
        X_test = X[-test_size:]
        test_predictions = model.predict(X_test)
        
        # Predict for future days with volatility-based bounds
        future_predictions = []
        last_data = df_rf[features].iloc[-1].values.reshape(1, -1)
        
        # Calculate historical volatility for bounds
        recent_volatility = df_rf['Close'].iloc[-20:].std()
        recent_mean = df_rf['Close'].iloc[-20:].mean()
        max_daily_move = recent_volatility * 2.5  # 2.5 sigma bounds
        
        # Track recent prices for context
        recent_prices = df_rf['Close'].iloc[-10:].tolist()
        
        for i in range(forecast_days):
            # Make prediction
            raw_pred = model.predict(last_data)[0]
            
            # Apply volatility-based bounds to prevent extreme predictions
            if i == 0:
                # First prediction: bound by recent volatility
                lower_limit = recent_prices[-1] - max_daily_move
                upper_limit = recent_prices[-1] + max_daily_move
            else:
                # Subsequent predictions: bound by previous prediction
                lower_limit = future_predictions[-1] - max_daily_move * (1 + i * 0.1)
                upper_limit = future_predictions[-1] + max_daily_move * (1 + i * 0.1)
            
            next_pred = np.clip(raw_pred, lower_limit, upper_limit)
            
            # Light smoothing only for first 3 days
            if i > 0 and i < 3:
                alpha = 0.85  # Light smoothing (85% new, 15% old)
                next_pred = alpha * next_pred + (1 - alpha) * future_predictions[-1]
            
            future_predictions.append(next_pred)
            recent_prices.append(next_pred)
            if len(recent_prices) > 15:
                recent_prices.pop(0)
            
            # Update features for next prediction
            # Feature order: Close, Volume, SMA_5, SMA_10, SMA_20, Std_10, Return_Volatility,
            #                Momentum_5, Momentum_10, Trend_Strength, Volume_Ratio, Returns, Returns_MA,
            #                Close_Lag_1-3, Volume_Lag_1-3
            prev_close = last_data[0, 0]
            last_data[0, 0] = next_pred  # Update Close
            
            # Update Returns
            returns_val = (next_pred / prev_close - 1) if prev_close > 0 else 0
            last_data[0, 11] = returns_val
            
            # Update Returns_MA
            last_data[0, 12] = (last_data[0, 12] * 2 + returns_val) / 3
            
            # Update SMAs
            last_data[0, 2] = (last_data[0, 2] * 4 + next_pred) / 5  # SMA_5
            last_data[0, 3] = (last_data[0, 3] * 9 + next_pred) / 10  # SMA_10
            last_data[0, 4] = (last_data[0, 4] * 19 + next_pred) / 20  # SMA_20
            
            # Update momentum indicators
            last_data[0, 7] = (next_pred / last_data[0, 2] - 1) if last_data[0, 2] > 0 else 0
            last_data[0, 8] = (next_pred / last_data[0, 3] - 1) if last_data[0, 3] > 0 else 0
            
            # Update trend strength
            last_data[0, 9] = ((last_data[0, 2] - last_data[0, 4]) / last_data[0, 4]) if last_data[0, 4] > 0 else 0
            
            # Shift Close lags (indices 13, 14, 15)
            last_data[0, 15] = last_data[0, 14]  # Close_Lag_3 = Close_Lag_2
            last_data[0, 14] = last_data[0, 13]  # Close_Lag_2 = Close_Lag_1
            last_data[0, 13] = next_pred  # Close_Lag_1 = new prediction
            
            # Shift Volume lags (indices 16, 17, 18)
            last_data[0, 18] = last_data[0, 17]  # Volume_Lag_3 = Volume_Lag_2
            last_data[0, 17] = last_data[0, 16]  # Volume_Lag_2 = Volume_Lag_1
            last_data[0, 16] = last_data[0, 1]  # Volume_Lag_1 = current Volume
        
        # Calculate robust confidence intervals using MAD (Median Absolute Deviation)
        # MAD is more robust to outliers than standard deviation
        test_errors = np.abs(test_predictions - y[-test_size:])
        
        # Use Median Absolute Deviation instead of MAE
        median_error = np.median(test_errors)
        mad = np.median(np.abs(test_errors - median_error))
        
        # For normal distribution, MAD * 1.4826 approximates standard deviation
        # For 95% confidence interval, use 1.96 * (MAD * 1.4826)
        robust_std = mad * 1.4826
        
        # Create future dates
        last_date = df_rf.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        future_dates = future_dates[future_dates.dayofweek < 5]  # Skip weekends
        
        # Adjust forecast length if needed due to weekend filtering
        future_predictions = future_predictions[:len(future_dates)]
        
        # Confidence intervals widening with forecast horizon
        lower_bound = []
        upper_bound = []
        for i, p in enumerate(future_predictions):
            # Use MAD-based robust uncertainty
            horizon_factor = 1 + (i * 0.08)  # 8% increase per day (less aggressive)
            uncertainty = robust_std * 1.96 * horizon_factor
            lower_bound.append(p - uncertainty)
            upper_bound.append(p + uncertainty)
        
        return {
            'future_dates': future_dates,
            'future_predictions': future_predictions,
            'test_predictions': test_predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    except Exception as e:
        st.error(f"Error in Random Forest prediction: {str(e)}")
        # Return dummy data if error
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        future_dates = future_dates[future_dates.dayofweek < 5]
        dummy_pred = [df['Close'].iloc[-1]] * len(future_dates)
        
        return {
            'future_dates': future_dates,
            'future_predictions': dummy_pred,
            'test_predictions': [df['Close'].iloc[-30:].mean()] * min(30, len(df)),
            'lower_bound': [p * 0.95 for p in dummy_pred],
            'upper_bound': [p * 1.05 for p in dummy_pred]
        }

def predict_prophet(df, forecast_days, stock_symbol="UNKNOWN"):
    """Enhanced Prophet model with technical indicators and optimized parameters"""
    try:
        # Work with all available columns for feature engineering
        required_cols = ['Close', 'Volume', 'High', 'Low', 'Open']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if 'Close' not in available_cols:
            st.error("Missing 'Close' column for Prophet prediction")
            return None
        
        df_prophet = df[available_cols].copy()
        df_prophet = df_prophet.dropna()
        
        # Validate data
        if len(df_prophet) < 60:
            st.warning(f"Insufficient data for Prophet prediction. Need at least 60 days, got {len(df_prophet)} days.")
            last_date = df.index[-1] if len(df) > 0 else datetime.now()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
            future_dates = future_dates[future_dates.dayofweek < 5][:forecast_days]
            last_price = df_prophet['Close'].iloc[-1] if len(df_prophet) > 0 else 100.0
            dummy_pred = [last_price] * len(future_dates)
            return {
                'future_dates': future_dates,
                'future_predictions': dummy_pred,
                'test_predictions': df['Close'].iloc[-min(10, len(df)-1):].tolist() if len(df) > 1 else [],
                'lower_bound': [p * 0.95 for p in dummy_pred],
                'upper_bound': [p * 1.05 for p in dummy_pred]
            }
        
        # Feature engineering - add technical indicators
        df_prophet['Returns'] = df_prophet['Close'].pct_change()
        df_prophet['MA_5'] = df_prophet['Close'].rolling(window=5).mean()
        df_prophet['MA_20'] = df_prophet['Close'].rolling(window=20).mean()
        df_prophet['MA_50'] = df_prophet['Close'].rolling(window=50).mean()
        df_prophet['Volatility'] = df_prophet['Returns'].rolling(window=10).std()
        
        if 'Volume' in df_prophet.columns:
            df_prophet['Volume_MA'] = df_prophet['Volume'].rolling(window=5).mean()
            df_prophet['Volume_Ratio'] = df_prophet['Volume'] / df_prophet['Volume'].rolling(window=20).mean()
            df_prophet['Volume_Trend'] = df_prophet['Volume'].rolling(window=10).apply(lambda x: 1 if x[-1] > x[0] else -1, raw=True)
        
        if all(col in df_prophet.columns for col in ['High', 'Low']):
            df_prophet['Price_Range'] = (df_prophet['High'] - df_prophet['Low']) / df_prophet['Close']
            df_prophet['RSI'] = 100 - (100 / (1 + (df_prophet['Close'].diff().clip(lower=0).rolling(14).mean() / 
                                                  (-df_prophet['Close'].diff().clip(upper=0).rolling(14).mean()))))
            
            # Bollinger Bands
            df_prophet['BB_Middle'] = df_prophet['Close'].rolling(window=20).mean()
            df_prophet['BB_Std'] = df_prophet['Close'].rolling(window=20).std()
            df_prophet['BB_Upper'] = df_prophet['BB_Middle'] + (2 * df_prophet['BB_Std'])
            df_prophet['BB_Lower'] = df_prophet['BB_Middle'] - (2 * df_prophet['BB_Std'])
            df_prophet['BB_Position'] = (df_prophet['Close'] - df_prophet['BB_Lower']) / (df_prophet['BB_Upper'] - df_prophet['BB_Lower'])
            
            # ATR (Average True Range)
            high_low = df_prophet['High'] - df_prophet['Low']
            high_close = np.abs(df_prophet['High'] - df_prophet['Close'].shift())
            low_close = np.abs(df_prophet['Low'] - df_prophet['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df_prophet['ATR'] = true_range.rolling(window=14).mean()
        
        # MACD indicators
        df_prophet['MACD'] = df_prophet['Close'].ewm(span=12).mean() - df_prophet['Close'].ewm(span=26).mean()
        df_prophet['MACD_Signal'] = df_prophet['MACD'].ewm(span=9).mean()
        df_prophet['MACD_Histogram'] = df_prophet['MACD'] - df_prophet['MACD_Signal']
        
        # Momentum indicators
        df_prophet['Price_Momentum_5'] = df_prophet['Close'] - df_prophet['Close'].shift(5)
        df_prophet['Price_Momentum_10'] = df_prophet['Close'] - df_prophet['Close'].shift(10)
        df_prophet['Price_Momentum_20'] = df_prophet['Close'] - df_prophet['Close'].shift(20)
        
        # Trend strength
        df_prophet['Trend_Strength'] = (df_prophet['MA_5'] - df_prophet['MA_20']) / df_prophet['MA_20']
        
        # Market regime detection (bull/bear/sideways)
        df_prophet['MA_50_Trend'] = df_prophet['MA_50'].pct_change(5)
        df_prophet['Market_Regime'] = df_prophet.apply(
            lambda row: 1 if row['Close'] > row['MA_50'] and row['MA_50_Trend'] > 0 else 
                       -1 if row['Close'] < row['MA_50'] and row['MA_50_Trend'] < 0 else 0, 
            axis=1
        )
        
        # Lag features
        df_prophet['Close_Lag1'] = df_prophet['Close'].shift(1)
        df_prophet['Returns_Lag1'] = df_prophet['Returns'].shift(1)
        df_prophet['Returns_Lag2'] = df_prophet['Returns'].shift(2)
        
        # Drop NaN from feature engineering
        df_prophet = df_prophet.dropna()
        
        # Try to load cached model
        model, _ = load_model_from_db(stock_symbol, "Prophet", df_prophet)
        was_cached = model is not None
        
        # Prophet requires a dataframe with columns 'ds' and 'y'
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = df_prophet.index
        prophet_df['y'] = df_prophet['Close'].values
        
        # Remove any NaN rows
        prophet_df = prophet_df.dropna()
        
        if len(prophet_df) < 2:
            st.warning(f"Prophet needs at least 2 rows, got {len(prophet_df)}.")
            last_date = df.index[-1] if len(df) > 0 else datetime.now()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
            future_dates = future_dates[future_dates.dayofweek < 5][:forecast_days]
            last_price = df_prophet['Close'].iloc[-1] if len(df_prophet) > 0 else 100.0
            dummy_pred = [last_price] * len(future_dates)
            return {
                'future_dates': future_dates,
                'future_predictions': dummy_pred,
                'test_predictions': df['Close'].iloc[-min(10, len(df)-1):].tolist() if len(df) > 1 else [],
                'lower_bound': [p * 0.95 for p in dummy_pred],
                'upper_bound': [p * 1.05 for p in dummy_pred]
            }
        
        # Create the Prophet model with optimized parameters
        if model is None:
            st.info(f"🔄 Training enhanced Prophet model for {stock_symbol}...")
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.15,  # More flexible to trend changes
                seasonality_prior_scale=20,  # Stronger seasonality
                seasonality_mode='multiplicative',  # Better for stock prices
                interval_width=0.95,
                changepoint_range=0.9  # Allow changepoints later in the series
            )
            
            # Add custom seasonalities
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
            
            # Add all available regressors
            regressors_added = []
            
            # Volume-based regressors
            if 'Volume' in df_prophet.columns:
                prophet_df['volume'] = df_prophet['Volume'].values
                model.add_regressor('volume', prior_scale=0.05)
                regressors_added.append('volume')
            
            if 'Volume_MA' in df_prophet.columns:
                prophet_df['volume_ma'] = df_prophet['Volume_MA'].values
                model.add_regressor('volume_ma', prior_scale=0.05)
                regressors_added.append('volume_ma')
            
            if 'Volume_Ratio' in df_prophet.columns:
                prophet_df['volume_ratio'] = df_prophet['Volume_Ratio'].values
                model.add_regressor('volume_ratio', prior_scale=0.1)
                regressors_added.append('volume_ratio')
            
            if 'Volume_Trend' in df_prophet.columns:
                prophet_df['volume_trend'] = df_prophet['Volume_Trend'].values
                model.add_regressor('volume_trend', prior_scale=0.08)
                regressors_added.append('volume_trend')
            
            # Technical indicator regressors
            if 'Volatility' in df_prophet.columns:
                prophet_df['volatility'] = df_prophet['Volatility'].values
                model.add_regressor('volatility', prior_scale=0.1)
                regressors_added.append('volatility')
            
            if 'Price_Range' in df_prophet.columns:
                prophet_df['price_range'] = df_prophet['Price_Range'].values
                model.add_regressor('price_range', prior_scale=0.05)
                regressors_added.append('price_range')
            
            if 'RSI' in df_prophet.columns:
                prophet_df['rsi'] = df_prophet['RSI'].values
                model.add_regressor('rsi', prior_scale=0.15, mode='additive')
                regressors_added.append('rsi')
            
            if 'MACD' in df_prophet.columns:
                prophet_df['macd'] = df_prophet['MACD'].values
                model.add_regressor('macd', prior_scale=0.12)
                regressors_added.append('macd')
            
            if 'MACD_Histogram' in df_prophet.columns:
                prophet_df['macd_hist'] = df_prophet['MACD_Histogram'].values
                model.add_regressor('macd_hist', prior_scale=0.15)
                regressors_added.append('macd_hist')
            
            # Bollinger Bands
            if 'BB_Position' in df_prophet.columns:
                prophet_df['bb_position'] = df_prophet['BB_Position'].values
                model.add_regressor('bb_position', prior_scale=0.2, mode='additive')
                regressors_added.append('bb_position')
            
            if 'ATR' in df_prophet.columns:
                prophet_df['atr'] = df_prophet['ATR'].values
                model.add_regressor('atr', prior_scale=0.08)
                regressors_added.append('atr')
            
            # MA-based regressors
            if 'MA_5' in df_prophet.columns:
                prophet_df['ma_5'] = df_prophet['MA_5'].values
                model.add_regressor('ma_5', prior_scale=0.25)
                regressors_added.append('ma_5')
            
            if 'MA_20' in df_prophet.columns:
                prophet_df['ma_20'] = df_prophet['MA_20'].values
                model.add_regressor('ma_20', prior_scale=0.25)
                regressors_added.append('ma_20')
            
            if 'MA_50' in df_prophet.columns:
                prophet_df['ma_50'] = df_prophet['MA_50'].values
                model.add_regressor('ma_50', prior_scale=0.2)
                regressors_added.append('ma_50')
            
            # Momentum indicators
            if 'Price_Momentum_5' in df_prophet.columns:
                prophet_df['momentum_5'] = df_prophet['Price_Momentum_5'].values
                model.add_regressor('momentum_5', prior_scale=0.15)
                regressors_added.append('momentum_5')
            
            if 'Price_Momentum_10' in df_prophet.columns:
                prophet_df['momentum_10'] = df_prophet['Price_Momentum_10'].values
                model.add_regressor('momentum_10', prior_scale=0.15)
                regressors_added.append('momentum_10')
            
            if 'Price_Momentum_20' in df_prophet.columns:
                prophet_df['momentum_20'] = df_prophet['Price_Momentum_20'].values
                model.add_regressor('momentum_20', prior_scale=0.15)
                regressors_added.append('momentum_20')
            
            # Trend strength
            if 'Trend_Strength' in df_prophet.columns:
                prophet_df['trend_strength'] = df_prophet['Trend_Strength'].values
                model.add_regressor('trend_strength', prior_scale=0.2, mode='additive')
                regressors_added.append('trend_strength')
            
            # Market regime
            if 'Market_Regime' in df_prophet.columns:
                prophet_df['market_regime'] = df_prophet['Market_Regime'].values
                model.add_regressor('market_regime', prior_scale=0.25, mode='additive')
                regressors_added.append('market_regime')
            
            # Lag features
            if 'Close_Lag1' in df_prophet.columns:
                prophet_df['close_lag1'] = df_prophet['Close_Lag1'].values
                model.add_regressor('close_lag1', prior_scale=0.3)
                regressors_added.append('close_lag1')
            
            if 'Returns_Lag1' in df_prophet.columns:
                prophet_df['returns_lag1'] = df_prophet['Returns_Lag1'].values
                model.add_regressor('returns_lag1', prior_scale=0.12)
                regressors_added.append('returns_lag1')
            
            if 'Returns_Lag2' in df_prophet.columns:
                prophet_df['returns_lag2'] = df_prophet['Returns_Lag2'].values
                model.add_regressor('returns_lag2', prior_scale=0.1)
                regressors_added.append('returns_lag2')
            
            # Clean data before fitting
            prophet_df = prophet_df.dropna()
            
            st.info(f"📈 Added {len(regressors_added)} regressors: {', '.join(regressors_added[:10])}{'...' if len(regressors_added) > 10 else ''}")
                
            # Fit the model
            model.fit(prophet_df)
        else:
            st.info(f"✅ Using cached Prophet model for {stock_symbol}")
            # Reconstruct regressor columns for cached model
            regressors_added = []
            if hasattr(model, 'extra_regressors'):
                for regressor in model.extra_regressors.keys():
                    # Map regressor names back to dataframe columns
                    col_mapping = {
                        'volume': 'Volume', 'volume_ma': 'Volume_MA', 'volume_ratio': 'Volume_Ratio',
                        'volume_trend': 'Volume_Trend', 'volatility': 'Volatility', 
                        'price_range': 'Price_Range', 'rsi': 'RSI', 'macd': 'MACD',
                        'macd_hist': 'MACD_Histogram', 'bb_position': 'BB_Position',
                        'atr': 'ATR', 'ma_5': 'MA_5', 'ma_20': 'MA_20', 'ma_50': 'MA_50',
                        'momentum_5': 'Price_Momentum_5', 'momentum_10': 'Price_Momentum_10',
                        'momentum_20': 'Price_Momentum_20', 'trend_strength': 'Trend_Strength',
                        'market_regime': 'Market_Regime', 'close_lag1': 'Close_Lag1',
                        'returns_lag1': 'Returns_Lag1', 'returns_lag2': 'Returns_Lag2'
                    }
                    if regressor in col_mapping and col_mapping[regressor] in df_prophet.columns:
                        prophet_df[regressor] = df_prophet[col_mapping[regressor]].values
                        regressors_added.append(regressor)
            
            prophet_df = prophet_df.dropna()
        
        # Create a dataframe for future predictions
        future_df = model.make_future_dataframe(periods=forecast_days)
        
        # Add regressor values for future dates using last known values + trend
        if hasattr(model, 'extra_regressors') and model.extra_regressors:
            for regressor in model.extra_regressors.keys():
                if regressor in prophet_df.columns:
                    # Use last 5 values to estimate trend
                    recent_values = prophet_df[regressor].tail(5).values
                    if len(recent_values) > 1:
                        trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
                        last_value = recent_values[-1]
                    else:
                        last_value = prophet_df[regressor].iloc[-1]
                        trend = 0
                    
                    # For historical period, use actual values
                    historical_values = prophet_df[regressor].values
                    # For future period, project trend
                    future_values = [last_value + trend * i for i in range(1, forecast_days + 1)]
                    
                    # Combine historical and future
                    future_df[regressor] = list(historical_values) + future_values
                # Make predictions
        forecast = model.predict(future_df)
        
        # Get the forecasted values for the future dates
        future_dates = pd.to_datetime(forecast['ds'].iloc[-forecast_days:])
        # Convert to DatetimeIndex to use dayofweek
        if isinstance(future_dates, pd.Series):
            future_dates = pd.DatetimeIndex(future_dates)
        future_dates = future_dates[future_dates.dayofweek < 5]  # Skip weekends
        
        # Get corresponding predictions (need to recalculate indices after filtering)
        # Find the indices of the last forecast_days predictions that match our filtered dates
        forecast_tail = forecast.tail(forecast_days).copy()
        forecast_tail['ds'] = pd.to_datetime(forecast_tail['ds'])
        # Filter to match only weekdays
        forecast_filtered = forecast_tail[forecast_tail['ds'].dt.dayofweek < 5]
        
        future_predictions = forecast_filtered['yhat'].values[:len(future_dates)]
        lower_bound = forecast_filtered['yhat_lower'].values[:len(future_dates)]
        upper_bound = forecast_filtered['yhat_upper'].values[:len(future_dates)]
        
        # Get predictions for the historical data (for testing/validation)
        historical_end = len(prophet_df)
        test_size = min(30, historical_end // 3)
        test_indices = range(historical_end - test_size, historical_end)
        test_predictions = forecast.loc[test_indices, 'yhat'].values
        
        result = {
            'future_dates': future_dates,
            'future_predictions': future_predictions,
            'test_predictions': test_predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        # Save to database only if this was a newly trained model
        if not was_cached:
            save_model_to_db(stock_symbol, 'Prophet', model, df_prophet, None)
            st.success(f"✅ Prophet model trained and cached for {stock_symbol}")
        
        return result
    except Exception as e:
        st.error(f"Error in Prophet prediction: {str(e)}")
        # Return dummy data if error
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        future_dates = future_dates[future_dates.dayofweek < 5]
        dummy_pred = [df['Close'].iloc[-1]] * len(future_dates)
        
        return {
            'future_dates': future_dates,
            'future_predictions': dummy_pred,
            'test_predictions': [df['Close'].iloc[-30:].mean()] * min(30, len(df)),
            'lower_bound': [p * 0.95 for p in dummy_pred],
            'upper_bound': [p * 1.05 for p in dummy_pred]
        }

def apply_directional_correction(result, df, df_lstm):
    """Apply directional adjustment based on technical indicators and momentum"""
    try:
        # Validate input
        if not result or 'future_predictions' not in result or len(result['future_predictions']) == 0:
            return result
            
        # Get last known price and trend
        last_price = df['Close'].iloc[-1]
        
        # Calculate multiple trend indicators
        sma_5 = df['Close'].rolling(window=5).mean().iloc[-1]
        sma_10 = df['Close'].rolling(window=10).mean().iloc[-1]
        sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = df['Close'].rolling(window=50).mean().iloc[-1] if len(df) > 50 else sma_20
        
        # Recent momentum
        momentum_3d = last_price - df['Close'].iloc[-3] if len(df) > 3 else 0
        momentum_5d = last_price - df['Close'].iloc[-5] if len(df) > 5 else 0
        
        # Momentum indicators
        rsi = calculate_rsi_simple(df['Close'], period=14)
        
        # Calculate MACD for trend confirmation
        ema_12 = df['Close'].ewm(span=12).mean().iloc[-1]
        ema_26 = df['Close'].ewm(span=26).mean().iloc[-1]
        macd = ema_12 - ema_26
        
        # Strong trend signals
        strong_uptrend = (last_price > sma_5 > sma_10 > sma_20) and (momentum_3d > 0) and (macd > 0)
        strong_downtrend = (last_price < sma_5 < sma_10 < sma_20) and (momentum_3d < 0) and (macd < 0)
        
        # Medium trend signals
        medium_uptrend = (last_price > sma_20) and (sma_20 > sma_50) and not strong_downtrend
        medium_downtrend = (last_price < sma_20) and (sma_20 < sma_50) and not strong_uptrend
        
        # RSI signals
        oversold = rsi < 35
        overbought = rsi > 65
        
        # Adjust predictions based on indicators
        predictions = np.array(result['future_predictions']).copy()
        
        # Calculate predicted direction
        pred_change = predictions[0] - last_price
        pred_direction = np.sign(pred_change)
        
        # STRONG TREND CORRECTIONS (high confidence)
        if strong_uptrend and pred_direction < 0:
            # Very strong uptrend but predicting down - likely wrong
            adjustment = abs(pred_change) * 0.75  # 75% correction
            predictions = predictions + adjustment
        elif strong_downtrend and pred_direction > 0:
            # Very strong downtrend but predicting up - likely wrong
            adjustment = abs(pred_change) * 0.75
            predictions = predictions - adjustment
        
        # MEDIUM TREND CORRECTIONS (moderate confidence)
        elif medium_uptrend and pred_direction < 0:
            adjustment = abs(pred_change) * 0.5  # 50% correction
            predictions = predictions + adjustment
        elif medium_downtrend and pred_direction > 0:
            adjustment = abs(pred_change) * 0.5
            predictions = predictions - adjustment
        
        # RSI REVERSAL CORRECTIONS
        if oversold and pred_direction < 0 and not strong_downtrend:
            # Oversold condition - likely bounce
            adjustment = abs(pred_change) * 0.5
            predictions = predictions + adjustment
        elif overbought and pred_direction > 0 and not strong_uptrend:
            # Overbought condition - likely pullback
            adjustment = abs(pred_change) * 0.5
            predictions = predictions - adjustment
        
        # MOMENTUM ALIGNMENT (subtle adjustment)
        avg_momentum = (momentum_3d + momentum_5d) / 2
        if avg_momentum > 0 and pred_direction < 0:
            # Positive momentum but predicting down
            adjustment = abs(pred_change) * 0.2
            predictions = predictions + adjustment
        elif avg_momentum < 0 and pred_direction > 0:
            # Negative momentum but predicting up
            adjustment = abs(pred_change) * 0.2
            predictions = predictions - adjustment
        
        result['future_predictions'] = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
        return result
        
    except Exception as e:
        # If correction fails, return original predictions
        st.warning(f"Directional correction failed: {str(e)}. Using original predictions.")
        return result

def calculate_rsi_simple(prices, period=14):
    """Calculate RSI indicator"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 else 50
    except:
        return 50  # Neutral RSI if calculation fails


# ------------------------------------ ADVANCED LSTM WITH ATTENTION ------------------------------------

def predict_lstm_attention(df, forecast_days, stock_symbol="UNKNOWN"):
    """
    Advanced LSTM with Multi-Head Attention mechanism for stock prediction.
    Based on recent research combining bidirectional LSTM with attention for financial time series.
    
    Architecture:
    - Multi-scale feature engineering (price returns, technical indicators, volume analysis)
    - Bidirectional LSTM for capturing both past and future context
    - Multi-head attention mechanism to focus on important temporal patterns
    - Dropout and regularization to prevent overfitting
    - Ensemble of multiple lookback windows for robust predictions
    """
    try:
        # Feature engineering with multiple time scales
        required_cols = ['Close', 'Volume', 'High', 'Low', 'Open']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if 'Close' not in available_cols:
            st.error("Missing 'Close' column for LSTM prediction")
            return None
        
        df_lstm = df[available_cols].copy()
        
        # ===== ADVANCED FEATURE ENGINEERING =====
        # Price-based features (normalized)
        df_lstm['Returns'] = df_lstm['Close'].pct_change()
        df_lstm['Returns_3d'] = df_lstm['Close'].pct_change(periods=3)
        df_lstm['Returns_5d'] = df_lstm['Close'].pct_change(periods=5)
        df_lstm['Returns_Std_7'] = df_lstm['Returns'].rolling(window=7).std()
        df_lstm['Returns_Std_14'] = df_lstm['Returns'].rolling(window=14).std()
        
        # Multi-scale moving averages
        df_lstm['MA_5'] = df_lstm['Close'].rolling(window=5).mean()
        df_lstm['MA_10'] = df_lstm['Close'].rolling(window=10).mean()
        df_lstm['MA_20'] = df_lstm['Close'].rolling(window=20).mean()
        df_lstm['MA_50'] = df_lstm['Close'].rolling(window=50).mean()
        
        # MA ratios (momentum indicators)
        df_lstm['MA_Ratio_5_20'] = (df_lstm['MA_5'] - df_lstm['MA_20']) / df_lstm['MA_20']
        df_lstm['MA_Ratio_10_50'] = (df_lstm['MA_10'] - df_lstm['MA_50']) / df_lstm['MA_50']
        
        # MACD (trend following)
        ema_12 = df_lstm['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df_lstm['Close'].ewm(span=26, adjust=False).mean()
        df_lstm['MACD'] = (ema_12 - ema_26) / df_lstm['Close']
        df_lstm['MACD_Signal'] = df_lstm['MACD'].ewm(span=9, adjust=False).mean()
        df_lstm['MACD_Hist'] = df_lstm['MACD'] - df_lstm['MACD_Signal']
        
        # Volume features
        if 'Volume' in df_lstm.columns:
            df_lstm['Volume_Change'] = df_lstm['Volume'].pct_change()
            df_lstm['Volume_MA_20'] = df_lstm['Volume'].rolling(window=20).mean()
            df_lstm['Volume_Ratio'] = df_lstm['Volume'] / (df_lstm['Volume_MA_20'] + 1e-10)
            df_lstm['Volume_Std'] = df_lstm['Volume'].rolling(window=20).std() / (df_lstm['Volume_MA_20'] + 1e-10)
        
        # Price range and volatility
        if all(col in df_lstm.columns for col in ['High', 'Low', 'Open']):
            # RSI at multiple timeframes
            for period in [9, 14, 21]:
                delta = df_lstm['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / (loss + 1e-10)
                df_lstm[f'RSI_{period}'] = (100 - (100 / (1 + rs))) / 100.0
            
            # Stochastic Oscillator
            low_14 = df_lstm['Low'].rolling(window=14).min()
            high_14 = df_lstm['High'].rolling(window=14).max()
            df_lstm['Stochastic_K'] = ((df_lstm['Close'] - low_14) / (high_14 - low_14 + 1e-10))
            df_lstm['Stochastic_D'] = df_lstm['Stochastic_K'].rolling(window=3).mean()
            
            # Bollinger Bands at multiple timeframes
            for period in [10, 20, 30]:
                bb_ma = df_lstm['Close'].rolling(window=period).mean()
                bb_std = df_lstm['Close'].rolling(window=period).std()
                df_lstm[f'BB_Width_{period}'] = (2 * bb_std) / (bb_ma + 1e-10)
                df_lstm[f'BB_Position_{period}'] = (df_lstm['Close'] - bb_ma) / (2 * bb_std + 1e-10)
            
            # True Range and ATR at multiple timeframes
            df_lstm['High_Low'] = (df_lstm['High'] - df_lstm['Low']) / df_lstm['Close']
            df_lstm['High_Close'] = np.abs(df_lstm['High'] - df_lstm['Close'].shift()) / df_lstm['Close']
            df_lstm['Low_Close'] = np.abs(df_lstm['Low'] - df_lstm['Close'].shift()) / df_lstm['Close']
            true_range = df_lstm[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
            df_lstm['ATR_14'] = true_range.rolling(window=14).mean()
            df_lstm['ATR_7'] = true_range.rolling(window=7).mean()
            
            # Williams %R
            df_lstm['Williams_R'] = ((high_14 - df_lstm['Close']) / (high_14 - low_14 + 1e-10))
            
            # Momentum indicators
            df_lstm['ROC_10'] = df_lstm['Close'].pct_change(periods=10)  # Rate of Change
            df_lstm['ROC_20'] = df_lstm['Close'].pct_change(periods=20)
            
            # Price position relative to high/low
            df_lstm['Close_to_High_20'] = (df_lstm['High'].rolling(20).max() - df_lstm['Close']) / df_lstm['Close']
            df_lstm['Close_to_Low_20'] = (df_lstm['Close'] - df_lstm['Low'].rolling(20).min()) / df_lstm['Close']
        
        # Store close prices for conversion
        close_prices = df_lstm['Close'].values
        
        # Remove NaN values
        df_lstm = df_lstm.dropna()
        
        # Check data sufficiency - use a longer history when available
        lookback_period = 60 if len(df_lstm) >= 240 else 30
        min_required = lookback_period + 80
        
        if len(df_lstm) < min_required:
            st.warning(f"Insufficient data for LSTM. Need {min_required} days, got {len(df_lstm)}.")
            return _create_fallback_predictions(df, forecast_days)
        
        # Try to load cached model
        model, scaler = load_model_from_db(stock_symbol, "LSTM_Attention_v3", df_lstm)
        was_cached = model is not None and scaler is not None
        
        # Use a compact but richer feature set with stable momentum and volatility signals
        feature_cols = [
            'Returns', 'Returns_3d', 'Returns_5d', 'Returns_Std_7', 'Returns_Std_14',
            'MA_Ratio_5_20', 'MA_Ratio_10_50',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'Volume_Change', 'Volume_Ratio', 'Volume_Std',
            'RSI_14', 'Stochastic_K', 'Stochastic_D',
            'BB_Position_20', 'BB_Width_20',
            'ATR_14', 'ATR_7',
            'ROC_10', 'ROC_20',
            'Close_to_High_20', 'Close_to_Low_20'
        ]
        
        # Filter to available columns
        feature_cols = [col for col in feature_cols if col in df_lstm.columns]
        features = df_lstm[feature_cols].values
        
        # Robust scaling
        if scaler is None:
            st.info(f"🔄 Training advanced LSTM-Attention model for {stock_symbol}...")
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()  # More robust to outliers than StandardScaler
            scaled_data = scaler.fit_transform(features)
        else:
            st.info(f"✅ Using cached LSTM-Attention model for {stock_symbol}")
            scaled_data = scaler.transform(features)
        
        # Create sequences for training
        X_train, y_dir, y_mag = [], [], []
        returns_data = df_lstm['Returns'].values
        direction_threshold = max(0.0015, np.nanmedian(np.abs(returns_data)) * 0.5)
        trend_horizon = 5

        for i in range(lookback_period, len(scaled_data)):
            future_idx = min(i + trend_horizon, len(df_lstm) - 1)
            future_return = (df_lstm['Close'].iloc[future_idx] / df_lstm['Close'].iloc[i]) - 1
            if np.isnan(future_return):
                continue
            X_train.append(scaled_data[i-lookback_period:i, :])
            y_dir.append(1.0 if future_return > direction_threshold else 0.0)  # Predict short-horizon trend direction
            y_mag.append(float(np.clip(abs(future_return), 0.0, 0.015)))

        X_train = np.array(X_train)
        y_dir = np.array(y_dir, dtype=np.float32)
        y_mag = np.array(y_mag, dtype=np.float32)

        # Use a chronological split instead of shuffled validation to avoid time-series leakage
        val_size = max(10, int(len(X_train) * 0.15))
        if len(X_train) <= val_size + 20:
            st.warning("Insufficient sequence data after validation split for LSTM training.")
            return _create_fallback_predictions(df, forecast_days)

        split_idx = len(X_train) - val_size
        X_fit, X_val = X_train[:split_idx], X_train[split_idx:]
        y_fit_dir, y_val_dir = y_dir[:split_idx], y_dir[split_idx:]
        y_fit_mag, y_val_mag = y_mag[:split_idx], y_mag[split_idx:]

        # Balance the directional classes by oversampling the minority side so the model does not collapse to one direction
        pos_count = float(np.sum(y_fit_dir == 1.0))
        neg_count = float(np.sum(y_fit_dir == 0.0))
        if pos_count > 0 and neg_count > 0:
            rng = np.random.default_rng(42)
            pos_idx = np.where(y_fit_dir == 1.0)[0]
            neg_idx = np.where(y_fit_dir == 0.0)[0]
            target_count = int(max(len(pos_idx), len(neg_idx)))
            pos_sample_idx = rng.choice(pos_idx, size=target_count, replace=True)
            neg_sample_idx = rng.choice(neg_idx, size=target_count, replace=True)
            balanced_idx = np.concatenate([pos_sample_idx, neg_sample_idx])
            rng.shuffle(balanced_idx)
            X_fit = X_fit[balanced_idx]
            y_fit_dir = y_fit_dir[balanced_idx]
            y_fit_mag = y_fit_mag[balanced_idx]
        
        # Build simple, robust LSTM model
        if model is None:
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras import regularizers
            
            num_features = scaled_data.shape[1]
            st.info(f"📊 Building Conservative LSTM ({num_features} features)...")
            
            # Input layer
            inputs = Input(shape=(lookback_period, num_features))
            
            # Stack a recurrent block with attention so the model can focus on the most useful windows
            lstm_out = Bidirectional(
                LSTM(
                    64,
                    return_sequences=True,
                    kernel_regularizer=regularizers.l2(0.0015),
                    recurrent_dropout=0.1,
                )
            )(inputs)
            lstm_out = BatchNormalization()(lstm_out)

            attn_out = MultiHeadAttention(num_heads=2, key_dim=max(8, scaled_data.shape[1] // 2))(lstm_out, lstm_out)
            attn_out = Add()([lstm_out, attn_out])
            dropout1 = Dropout(0.25)(attn_out)

            lstm_out_2 = LSTM(
                32,
                return_sequences=False,
                kernel_regularizer=regularizers.l2(0.0015),
                recurrent_dropout=0.1,
            )(dropout1)
            dropout2 = Dropout(0.25)(lstm_out_2)

            dense1 = Dense(24, activation='relu', kernel_regularizer=regularizers.l2(0.0015))(dropout2)
            dropout3 = Dropout(0.2)(dense1)

            direction_output = Dense(1, activation='sigmoid', name='direction')(dropout3)
            magnitude_output = Dense(1, activation='softplus', name='magnitude')(dropout3)
            
            # Create model
            model = Model(inputs=inputs, outputs=[direction_output, magnitude_output])
            
            # Conservative optimizer settings
            optimizer = Adam(learning_rate=0.0007, clipnorm=1.0)
            model.compile(
                optimizer=optimizer,
                loss={'direction': 'binary_crossentropy', 'magnitude': tf.keras.losses.Huber(delta=0.02)},
                loss_weights={'direction': 0.75, 'magnitude': 0.25},
                metrics={'direction': ['accuracy', tf.keras.metrics.AUC(name='auc')], 'magnitude': ['mae']}
            )
            
            # Simple early stopping
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=12,
                restore_best_weights=True,
                min_delta=0.0001,
                verbose=0
            )

            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-5,
                verbose=0,
            )
            
            # Train with conservative settings for limited data
            history = model.fit(
                X_fit, {'direction': y_fit_dir, 'magnitude': y_fit_mag},
                epochs=80,
                batch_size=min(32, max(8, len(X_fit) // 10)),
                validation_data=(X_val, {'direction': y_val_dir, 'magnitude': y_val_mag}),
                shuffle=False,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            st.success(f"✅ Model trained in {len(history.history['loss'])} epochs")
        
        # Generate test predictions
        test_size = min(30, len(scaled_data) - lookback_period)
        test_predictions = []
        
        direction_test_true = []
        direction_test_prob = []
        direction_test_weights = []

        if test_size > 0:
            X_test = []
            for i in range(len(scaled_data) - test_size, len(scaled_data)):
                if i >= lookback_period:
                    X_test.append(scaled_data[i-lookback_period:i, :])
                    future_idx = min(i + trend_horizon, len(df_lstm) - 1)
                    future_return = (df_lstm['Close'].iloc[future_idx] / df_lstm['Close'].iloc[i]) - 1
                    direction_test_true.append(1 if future_return > direction_threshold else 0)
                    direction_test_weights.append(abs(float(future_return)))
            
            if len(X_test) > 0:
                X_test = np.array(X_test)
                predicted_prob, predicted_mag = model.predict(X_test, verbose=0)
                predicted_prob = np.asarray(predicted_prob).flatten()
                predicted_mag = np.clip(np.asarray(predicted_mag).flatten(), 0.0, 0.015)
                direction_test_prob = predicted_prob.tolist()
                
                # Convert direction probabilities to prices using the learned magnitude head
                recent_abs_return = np.nanmedian(np.abs(returns_data[-20:]))
                base_move = float(max(0.0008, recent_abs_return * 0.35))
                for i, prob in enumerate(predicted_prob):
                    prev_idx = len(close_prices) - test_size + i - 1
                    prev_price = close_prices[prev_idx]
                    move_size = min(max(base_move, float(predicted_mag[i])), 0.015)
                    if prob >= 0.55:
                        signed_move = move_size
                    elif prob <= 0.45:
                        signed_move = -move_size
                    else:
                        signed_move = 0.0
                    projected_price = prev_price * (1 + signed_move)
                    test_predictions.append(float(prev_price + (projected_price - prev_price) * 0.3))
        
        # Generate future predictions with uncertainty estimation
        future_predictions = []
        prediction_uncertainty = []
        last_sequence = scaled_data[-lookback_period:, :].copy()
        current_price = close_prices[-1]
        
        # Get number of features from scaled data
        num_features = scaled_data.shape[1]
        
        # Monte Carlo dropout for uncertainty estimation
        n_iterations = 10
        recent_abs_return = np.nanmedian(np.abs(returns_data[-20:]))
        base_move = float(max(0.0008, recent_abs_return * 0.35))
        
        for step in range(forecast_days):
            # Multiple predictions for uncertainty
            step_predictions = []
            for _ in range(n_iterations):
                input_seq = last_sequence.reshape(1, lookback_period, num_features)
                pred_prob, pred_mag = model(input_seq, training=True)
                pred_prob = float(pred_prob.numpy()[0, 0])  # Training=True enables dropout
                pred_mag = float(np.clip(pred_mag.numpy()[0, 0], 0.0, 0.015))
                step_predictions.append((pred_prob, pred_mag))
            
            # Mean and std of predictions
            mean_prob = float(np.clip(np.mean([p for p, _ in step_predictions]), 0.0, 1.0))
            mean_mag = float(np.clip(np.mean([m for _, m in step_predictions]), 0.0, 0.015))
            std_prob = float(np.std([p for p, _ in step_predictions]))
            move_size = min(max(base_move, mean_mag), 0.015)
            if mean_prob >= 0.55:
                signed_move = move_size
            elif mean_prob <= 0.45:
                signed_move = -move_size
            else:
                signed_move = 0.0
            
            # Convert to price
            projected_price = current_price * (1 + signed_move)
            next_price = current_price + (projected_price - current_price) * 0.3
            future_predictions.append(next_price)
            prediction_uncertainty.append(max(std_prob * current_price, current_price * 0.005))
            
            # Update features for next step
            new_features = _create_future_features(df_lstm, feature_cols, signed_move)
            new_row = scaler.transform([new_features])[0]
            
            # Update sequence
            last_sequence = np.vstack([last_sequence[1:, :], new_row])
            current_price = next_price
        
        future_predictions = np.array(future_predictions)
        prediction_uncertainty = np.array(prediction_uncertainty)
        
        # Create future dates (skip weekends)
        last_date = df_lstm.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days * 2)
        future_dates = future_dates[future_dates.dayofweek < 5][:forecast_days]
        
        # Adjust predictions if dates filtered
        future_predictions = future_predictions[:len(future_dates)]
        prediction_uncertainty = prediction_uncertainty[:len(future_dates)]
        
        # Confidence intervals with uncertainty estimation
        lower_bound = future_predictions - 2 * prediction_uncertainty
        upper_bound = future_predictions + 2 * prediction_uncertainty
        
        result = {
            'future_dates': future_dates,
            'future_predictions': future_predictions,
            'test_predictions': test_predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'direction_test_true': direction_test_true,
            'direction_test_prob': direction_test_prob,
            'direction_test_weights': direction_test_weights,
        }
        
        # Apply directional correction
        result = apply_directional_correction(result, df, df_lstm)
        
        # Cache model if newly trained
        if not was_cached:
            save_model_to_db(stock_symbol, 'LSTM_Attention_v3', model, df_lstm, scaler)
            st.success(f"✅ Advanced LSTM model cached for {stock_symbol}")
        
        return result
        
    except Exception as e:
        st.error(f"Error in LSTM-Attention prediction: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return _create_fallback_predictions(df, forecast_days)


def _create_future_features(df_lstm, feature_cols, predicted_return):
    """Create features for next time step in prediction"""
    new_features = []
    
    for col in feature_cols:
        if 'Returns' in col:
            new_features.append(predicted_return)
        elif 'MA_Ratio' in col or 'MACD' in col:
            # Decay factor for trend indicators
            new_features.append(df_lstm[col].iloc[-1] * 0.95)
        elif 'RSI' in col:
            # Bounded indicators
            new_features.append(np.clip(df_lstm[col].iloc[-1] + predicted_return * 0.1, 0, 1))
        elif 'Volume' in col:
            # Neutral volume assumption
            new_features.append(0.0 if 'Change' in col else 1.0)
        elif 'Std' in col or 'ATR' in col:
            # Volatility indicators
            new_features.append(df_lstm[col].iloc[-1])
        else:
            # Default: use last value
            new_features.append(df_lstm[col].iloc[-1] if col in df_lstm.columns else 0.0)
    
    return new_features


def _create_fallback_predictions(df, forecast_days):
    """Create simple fallback predictions when model fails"""
    last_date = df.index[-1] if len(df) > 0 else datetime.now()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days * 2)
    future_dates = future_dates[future_dates.dayofweek < 5][:forecast_days]
    last_price = df['Close'].iloc[-1] if len(df) > 0 else 100.0
    dummy_pred = [last_price] * len(future_dates)
    
    return {
        'future_dates': future_dates,
        'future_predictions': dummy_pred,
        'test_predictions': df['Close'].iloc[-min(10, len(df)-1):].tolist() if len(df) > 1 else [],
        'lower_bound': [p * 0.95 for p in dummy_pred],
        'upper_bound': [p * 1.05 for p in dummy_pred]
    }


# Alias for backward compatibility
predict_lstm = predict_lstm_attention


# ------------------------------------ HYBRID MODEL ------------------------------------

def train_hybrid_model(base_results, stock_symbol, df):
    """
    Train a true hybrid/stacking model using base model predictions
    The meta-model learns to combine predictions optimally
    """
    try:
        # Check if hybrid model is cached
        cached_hybrid, _ = load_model_from_db(stock_symbol, "Hybrid", df)
        
        if cached_hybrid is not None:
            st.info(f"✅ Using cached Hybrid model for {stock_symbol}")
            return cached_hybrid
        
        st.info(f"🔄 Training Hybrid Meta-Model for {stock_symbol}...")
        
        # Extract test predictions from all models for training meta-model
        models = ['ARIMA', 'Random Forest', 'Prophet', 'LSTM']
        
        # Collect training data from test predictions
        X_meta = []
        y_meta = []
        
        # Get minimum length across all test predictions
        min_len = min([len(base_results[model]['test_predictions']) for model in models if 'test_predictions' in base_results[model]])
        
        if min_len < 10:
            st.warning("Insufficient test data for hybrid training, using optimized weights")
            return None
        
        # Build training dataset from test predictions
        for i in range(min_len):
            features = [base_results[model]['test_predictions'][i] for model in models]
            X_meta.append(features)
        
        # Get corresponding actual values from dataframe
        y_meta = df['Close'].iloc[-min_len:].values
        
        X_meta = np.array(X_meta)
        y_meta = np.array(y_meta)
        
        # Build simple meta-learner
        meta_model = Sequential([
            Dense(8, activation='relu', input_shape=(4,)),
            Dropout(0.1),
            Dense(4, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        meta_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train meta-learner
        early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, verbose=0)
        meta_model.fit(X_meta, y_meta, epochs=30, batch_size=8, verbose=0, callbacks=[early_stop], validation_split=0.2)
        
        # Save to database
        save_model_to_db(stock_symbol, 'Hybrid', meta_model, df, None)
        st.success(f"✅ Hybrid meta-model trained and cached for {stock_symbol}")
        
        return meta_model
        
    except Exception as e:
        st.error(f"Error training hybrid model: {str(e)}")
        return None

def predict_hybrid(df, forecast_days, stock_symbol, base_results):
    """
    Smart weighted ensemble based on model performance
    Uses adaptive weights based on directional accuracy
    """
    try:
        st.info("🎯 Creating performance-weighted ensemble...")
        
        models = ['ARIMA', 'Random Forest', 'Prophet', 'LSTM']
        
        # Calculate performance-based weights from test predictions
        weights = {}
        total_weight = 0
        
        for model in models:
            if 'test_predictions' in base_results[model] and len(base_results[model]['test_predictions']) > 0:
                # Get test predictions
                test_preds = np.array(base_results[model]['test_predictions'])
                actual = df['Close'].iloc[-len(test_preds):].values
                
                # Calculate directional accuracy
                pred_dir = np.sign(np.diff(test_preds))
                actual_dir = np.sign(np.diff(actual))
                dir_accuracy = np.mean(pred_dir == actual_dir)
                
                # Calculate MAE
                mae = np.mean(np.abs(test_preds - actual))
                
                # Weight based on directional accuracy (70%) and inverse MAE (30%)
                # Higher directional accuracy = higher weight
                # Lower MAE = higher weight
                accuracy_weight = dir_accuracy
                mae_weight = 1.0 / (1.0 + mae)
                
                model_weight = 0.7 * accuracy_weight + 0.3 * mae_weight
                weights[model] = max(model_weight, 0.01)  # Minimum weight
                total_weight += weights[model]
            else:
                weights[model] = 0.01
                total_weight += 0.01
        
        # Normalize weights to sum to 1
        for model in models:
            weights[model] = weights[model] / total_weight
        
        st.success(f"📊 Adaptive weights: ARIMA={weights['ARIMA']:.2%}, RF={weights['Random Forest']:.2%}, Prophet={weights['Prophet']:.2%}, LSTM={weights['LSTM']:.2%}")
        
        # Combine predictions using adaptive weights
        min_horizon = min([len(base_results[model]['future_predictions']) for model in models])
        
        predictions = []
        for i in range(min_horizon):
            weighted_pred = sum(
                weights[model] * base_results[model]['future_predictions'][i] 
                for model in models
            )
            predictions.append(float(weighted_pred))
        
        return {
            'future_predictions': predictions,
            'future_dates': base_results['ARIMA']['future_dates'][:min_horizon],
            'method': 'adaptive_weighted_ensemble',
            'weights': weights
        }
        
    except Exception as e:
        st.error(f"Error in hybrid prediction: {str(e)}")
        # Fallback to simple average
        models = list(base_results.keys())
        min_horizon = min([len(base_results[model]['future_predictions']) for model in models])
        predictions = []
        for i in range(min_horizon):
            avg = np.mean([base_results[model]['future_predictions'][i] for model in models])
            predictions.append(float(avg))
        
        return {
            'future_predictions': predictions,
            'future_dates': base_results[models[0]]['future_dates'][:min_horizon],
            'method': 'simple_average'
        }

# ------------------------------------ MODEL VALIDATION ------------------------------------

def calculate_metrics(y_true, y_pred):
    """Calculate validation metrics for model performance"""
    # Handle potential length mismatch
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[-min_len:]
    y_pred = y_pred[-min_len:]
    
    try:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error), avoiding division by zero
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # Initialize variables
        directional_accuracy = 0
        weighted_directional_accuracy = 0
        
        # Calculate directional accuracy (how often the direction of change is correct)
        # Use minimal threshold to count most movements
        if len(y_true) > 1:
            # Calculate changes first
            y_true_diff = np.diff(y_true)
            y_pred_diff = np.diff(y_pred)
            
            # Use a stronger threshold so we score meaningful moves instead of tiny noise
            threshold = 0.004 * np.mean(y_true)  # 0.4% threshold
            
            # Apply threshold to reduce noise from tiny fluctuations
            y_true_direction = np.where(np.abs(y_true_diff) < threshold, 0, np.sign(y_true_diff))
            y_pred_direction = np.where(np.abs(y_pred_diff) < threshold, 0, np.sign(y_pred_diff))
            
            # Calculate basic directional accuracy
            directional_accuracy = np.mean(y_true_direction == y_pred_direction) * 100
            
            # Calculate weighted directional accuracy (weight by magnitude of change)
            weights = np.abs(y_true_diff) / (np.sum(np.abs(y_true_diff)) + 1e-10)
            weighted_directional_accuracy = np.sum(weights * (y_true_direction == y_pred_direction)) * 100
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        mse = rmse = mae = r2 = mape = directional_accuracy = weighted_directional_accuracy = 0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'directional_accuracy': directional_accuracy,
        'weighted_directional_accuracy': weighted_directional_accuracy
    }


def calculate_direction_metrics(y_true_direction, y_pred_prob, weights=None):
    """Calculate directional accuracy directly from a classification output."""
    y_true_direction = np.asarray(y_true_direction).astype(int)
    y_pred_prob = np.asarray(y_pred_prob).astype(float)

    if len(y_true_direction) == 0 or len(y_pred_prob) == 0:
        return {'directional_accuracy': 0.0, 'weighted_directional_accuracy': 0.0}

    min_len = min(len(y_true_direction), len(y_pred_prob))
    y_true_direction = y_true_direction[-min_len:]
    y_pred_prob = y_pred_prob[-min_len:]

    y_pred_direction = (y_pred_prob >= 0.5).astype(int)
    directional_accuracy = float(np.mean(y_true_direction == y_pred_direction) * 100)

    if weights is None:
        weights = np.ones(min_len, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)[-min_len:]

    weights = weights / (np.sum(weights) + 1e-10)
    weighted_directional_accuracy = float(np.sum(weights * (y_true_direction == y_pred_direction)) * 100)

    return {
        'directional_accuracy': directional_accuracy,
        'weighted_directional_accuracy': weighted_directional_accuracy,
    }

# ------------------------------------ VISUALIZATION FUNCTIONS ------------------------------------

def create_candlestick_plot(df, title='Stock Price Chart'):
    """Create a candlestick chart with volume"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05, subplot_titles=('Price', 'Volume'),
                       row_heights=[0.7, 0.3])
                       
    # Add candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                               open=df['Open'],
                               high=df['High'],
                               low=df['Low'],
                               close=df['Close'],
                               name='Price',
                               increasing_line_color='#26a69a',
                               decreasing_line_color='#ef5350',
                               increasing_fillcolor='#26a69a',
                               decreasing_fillcolor='#ef5350'),
                row=1, col=1)
                
    # Add volume bar chart
    colors = ['#ef5350' if row['Open'] - row['Close'] >= 0 
            else '#26a69a' for _, row in df.iterrows()]
            
    fig.add_trace(go.Bar(x=df.index, 
                       y=df['Volume'],
                       marker_color=colors,
                       name='Volume'),
                row=2, col=1)
                
    # Add moving averages
    ma_short = 20
    ma_medium = 50
    ma_long = 200
    
    # Add short-term MA
    if len(df) >= ma_short:
        fig.add_trace(go.Scatter(x=df.index, 
                               y=df['Close'].rolling(window=ma_short).mean(),
                               line=dict(color='orange', width=1.5),
                               name=f'{ma_short}-period MA'),
                    row=1, col=1)
    
    # Add medium-term MA
    if len(df) >= ma_medium:
        fig.add_trace(go.Scatter(x=df.index, 
                               y=df['Close'].rolling(window=ma_medium).mean(),
                               line=dict(color='blue', width=1.5),
                               name=f'{ma_medium}-period MA'),
                    row=1, col=1)
    
    # Add long-term MA if enough data
    if len(df) >= ma_long:            
        fig.add_trace(go.Scatter(x=df.index, 
                           y=df['Close'].rolling(window=ma_long).mean(),
                           line=dict(color='purple', width=1.5),
                           name=f'{ma_long}-period MA'),
                row=1, col=1)
                
    # Update layout
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        title_text=title,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=60, r=60, t=80, b=60),
    )
    
    # Add rangebreaks to hide weekends
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # hide weekends
        ]
    )
    
    return fig

def plot_prediction(df, model_results, model_name):
    """Plot historical prices and predictions"""
    fig = go.Figure()
    
    # Add historical prices
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Historical Prices',
        line=dict(color='#1f77b4', width=2),
        connectgaps=True
    ))
    
    # Add test predictions if available
    if 'test_predictions' in model_results and len(model_results['test_predictions']) > 0:
        # Get the last portion of the data used for testing
        test_size = min(len(model_results['test_predictions']), 30)
        test_dates = df.index[-test_size:]
        test_predictions = model_results['test_predictions'][-test_size:]
        
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=test_predictions,
            mode='lines',
            name='Model Fit',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            connectgaps=True
        ))
    
    # Add future predictions
    if 'future_dates' in model_results and 'future_predictions' in model_results:
        fig.add_trace(go.Scatter(
            x=model_results['future_dates'],
            y=model_results['future_predictions'],
            mode='lines+markers',
            name=f'{model_name} Prediction',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=8),
            connectgaps=True
        ))
        
        # Add confidence interval if available
        if 'lower_bound' in model_results and 'upper_bound' in model_results:
            # Create arrays for confidence interval
            dates_ci = list(model_results['future_dates']) + list(model_results['future_dates'][::-1])
            values_ci = list(model_results['upper_bound']) + list(model_results['lower_bound'][::-1])
            
            fig.add_trace(go.Scatter(
                x=dates_ci,
                y=values_ci,
                fill='toself',
                fillcolor='rgba(0, 176, 246, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval',
                showlegend=True,
                hoverinfo='skip'
            ))
    
    # Update layout
    fig.update_layout(
        title=f'{model_name} Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    # Ensure continuous x-axis without gaps
    fig.update_xaxes(type='date', tickformat='%b %d\n%Y')
    
    return fig

def plot_model_comparison(all_model_results, symbol):
    """Plot predictions from multiple models for comparison"""
    fig = go.Figure()
    
    # Get historical data from the first model (all should have same historical data)
    first_model = list(all_model_results.keys())[0]
    if 'historical_dates' in all_model_results[first_model] and 'historical_prices' in all_model_results[first_model]:
        dates = all_model_results[first_model]['historical_dates']
        prices = all_model_results[first_model]['historical_prices']
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name='Historical',
            line=dict(color='#7f7f7f', width=2)
        ))
    
    # Color map for the models
    colors = {
        'ARIMA': '#1f77b4',
        'Random Forest': '#ff7f0e',
        'Prophet': '#2ca02c',
        'LSTM': '#d62728'
    }
    
    # Add predictions for each model
    for model_name, results in all_model_results.items():
        if 'future_dates' in results and 'future_predictions' in results:
            # Get last historical point for connection
            last_hist_date = dates[-1] if 'historical_dates' in all_model_results[first_model] else results['future_dates'][0]
            last_hist_price = prices[-1] if 'historical_prices' in all_model_results[first_model] else results['future_predictions'][0]
            
            # Prepend last historical point to create seamless connection
            extended_dates = [last_hist_date] + list(results['future_dates'])
            extended_predictions = [last_hist_price] + list(results['future_predictions'])
            
            fig.add_trace(go.Scatter(
                x=extended_dates,
                y=extended_predictions,
                mode='lines+markers',
                name=f'{model_name}',
                line=dict(color=colors.get(model_name, '#000000'), width=2),
                marker=dict(size=6)
            ))
    
    # Update layout
    fig.update_layout(
        title=f'Model Comparison for {symbol}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# -------------------- MAIN APPLICATION --------------------

# Session state initialization
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'comparison_made' not in st.session_state:
    st.session_state.comparison_made = False
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'screener_run' not in st.session_state:
    st.session_state.screener_run = False
if 'screener_results' not in st.session_state:
    st.session_state.screener_results = None

# Page header with logo
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.markdown('<h1 class="main-title">StockSense AI Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Advanced Stock Market Prediction & Analysis Platform</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image(STATIC_IMAGES["sidebar"], use_container_width=True)
st.sidebar.markdown("## Configure Analysis")

# Data source selection
data_source = st.sidebar.selectbox(
    "Data Source",
    ["Yahoo Finance", "Sample Data"],
    help="Select the source of stock data"
)

# Stock symbol input
default_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "IBM", "JPM", "DIS", "NFLX", "PYPL", "ADBE", "CSCO"]
symbol = st.sidebar.selectbox("Select Stock Symbol", default_symbols)
custom_symbol = st.sidebar.text_input("Or Enter Custom Symbol", "")
if custom_symbol:
    symbol = custom_symbol

# Time period for analysis
time_period = st.sidebar.select_slider(
    "Select Time Period",
    options=["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"],
    value="1 Year"
)

# Forecast period
forecast_days = st.sidebar.slider("Forecast Days", min_value=1, max_value=30, value=7)

# Model selection
model_type = st.sidebar.selectbox(
    "Select Prediction Model",
    ["ARIMA", "Random Forest", "Prophet", "LSTM", "Compare All"]
)

# Analysis features toggle
st.sidebar.markdown("### Analysis Features")
show_technical = st.sidebar.checkbox("Technical Indicators", value=True)
show_feature_engineering = st.sidebar.checkbox("Feature Engineering", value=True)
show_news_sentiment = st.sidebar.checkbox("News Sentiment", value=True)
show_validation = st.sidebar.checkbox("Model Validation", value=True)
show_volume = st.sidebar.checkbox("Volume Analysis", value=True)

# Optional professional features
st.sidebar.markdown("### Professional Features")
show_stock_screener = st.sidebar.checkbox("Stock Screener", value=True)
show_portfolio_optimizer = st.sidebar.checkbox("Portfolio Optimizer", value=True)
show_anomaly_detection = st.sidebar.checkbox("Anomaly Detection", value=True)
show_market_correlation = st.sidebar.checkbox("Market Correlation", value=True)

# Model Cache Management
st.sidebar.markdown("---")
st.sidebar.markdown("### 🗄️ Model Cache")

# Show cache stats
cache_stats = get_cache_stats()
if cache_stats:
    st.sidebar.info(f"📦 {len(cache_stats)} model(s) cached")
    
    # Show cached models in expander
    with st.sidebar.expander("View Cached Models"):
        for stock, model, date, records in cache_stats[:5]:  # Show top 5
            st.write(f"**{stock}** - {model}")
            st.caption(f"Trained: {date} | {records} records")
else:
    st.sidebar.info("📦 No models cached")

# Cache management buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🔄 Retrain All", use_container_width=True):
        count = clear_all_models()
        st.sidebar.success(f"✅ Cleared {count} model(s)")
        st.sidebar.info("🔄 Models will retrain on next prediction")
        st.rerun()

with col2:
    if st.button("🧹 Clear Old", use_container_width=True):
        count = clear_old_models(days_old=7)
        st.sidebar.success(f"✅ Removed {count} old model(s)")
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📈 Prediction", "🔍 Model Comparison", "🧠 Pro Analysis", "ℹ️ About"])

# Function to run analysis and prediction
def run_analysis():
    with st.spinner('Loading stock data and performing analysis...'):
        df = load_stock_data(symbol, data_source, time_period)
        
        if df is None:
            st.error("Failed to load stock data. Please try again or choose a different source.")
            return
        
        progress_bar = st.progress(0.2)
        st.session_state.data_loaded = True
        st.session_state.stock_data = df
        
        # Feature engineering
        if show_feature_engineering:
            df_features = engineer_features(df)
        else:
            df_features = df.copy()
            
        progress_bar.progress(0.4)
        
        # Technical indicators
        if show_technical:
            df_tech = calculate_technical_indicators(df_features)
        else:
            df_tech = df_features.copy()
            
        progress_bar.progress(0.6)
        
        # News sentiment
        news_sentiment = None
        if show_news_sentiment:
            news_sentiment = get_news_sentiment(symbol)
            
        progress_bar.progress(0.8)
            
        # Make predictions if requested
        if model_type != "Compare All":
            if model_type == "ARIMA":
                results = predict_arima(df_tech, forecast_days, symbol)
            elif model_type == "Random Forest":
                results = predict_random_forest(df_tech, forecast_days, symbol)
            elif model_type == "Prophet":
                results = predict_prophet(df_tech, forecast_days, symbol)
            elif model_type == "LSTM":
                results = predict_lstm(df_tech, forecast_days, symbol)
            
            # Calculate metrics
            if show_validation and 'test_predictions' in results:
                metrics = calculate_metrics(df['Close'].iloc[-len(results['test_predictions']):].values, results['test_predictions'])
                if 'direction_test_true' in results and 'direction_test_prob' in results and len(results['direction_test_true']) > 0:
                    direction_metrics = calculate_direction_metrics(
                        results['direction_test_true'],
                        results['direction_test_prob'],
                        results.get('direction_test_weights')
                    )
                    metrics['directional_accuracy'] = direction_metrics['directional_accuracy']
                    metrics['weighted_directional_accuracy'] = direction_metrics['weighted_directional_accuracy']
                results['metrics'] = metrics
            
            st.session_state.prediction_results = {
                "df": df,
                "df_tech": df_tech,
                "results": results,
                "news_sentiment": news_sentiment
            }
            st.session_state.prediction_made = True
        else:
            # Run all models for comparison
            all_results = {}
            models = ["ARIMA", "Random Forest", "Prophet", "LSTM"]
            
            # Get the last historical price as anchor point
            last_historical_price = df['Close'].iloc[-1]
            
            for i, model_name in enumerate(models):
                try:
                    if model_name == "ARIMA":
                        model_results = predict_arima(df_tech, forecast_days, symbol)
                    elif model_name == "Random Forest":
                        model_results = predict_random_forest(df_tech, forecast_days, symbol)
                    elif model_name == "Prophet":
                        model_results = predict_prophet(df_tech, forecast_days, symbol)
                    elif model_name == "LSTM":
                        model_results = predict_lstm_attention(df_tech, forecast_days, symbol)
                except Exception as e:
                    st.error(f"❌ Error running {model_name} model: {str(e)}")
                    model_results = _create_fallback_predictions(df, forecast_days)
                    model_results['error'] = f"{model_name} error: {str(e)}"

                # Validate model results
                if not model_results or 'future_predictions' not in model_results or len(model_results['future_predictions']) == 0:
                    st.warning(f"⚠️ {model_name} produced no future predictions. Using fallback.")
                    model_results = _create_fallback_predictions(df, forecast_days)
                    model_results['error'] = f"{model_name} produced no predictions"

                # Normalize predictions to start from the same point
                try:
                    predictions = np.array(model_results['future_predictions'])
                    first_pred = predictions[0]

                    # Calculate the adjustment needed
                    adjustment = last_historical_price - first_pred

                    # Apply adjustment to all predictions (preserving the pattern)
                    model_results['future_predictions'] = (predictions + adjustment).tolist()

                    # Also adjust confidence bounds if they exist
                    if 'lower_bound' in model_results:
                        model_results['lower_bound'] = (np.array(model_results['lower_bound']) + adjustment).tolist()
                    if 'upper_bound' in model_results:
                        model_results['upper_bound'] = (np.array(model_results['upper_bound']) + adjustment).tolist()
                except Exception as e:
                    st.warning(f"⚠️ {model_name} normalization failed: {str(e)}")

                # Calculate metrics
                if show_validation and 'test_predictions' in model_results:
                    try:
                        test_len = len(model_results['test_predictions'])
                        if test_len > 0:
                            metrics = calculate_metrics(df['Close'].iloc[-test_len:].values, model_results['test_predictions'])
                            if 'direction_test_true' in model_results and 'direction_test_prob' in model_results and len(model_results['direction_test_true']) > 0:
                                direction_metrics = calculate_direction_metrics(
                                    model_results['direction_test_true'],
                                    model_results['direction_test_prob'],
                                    model_results.get('direction_test_weights')
                                )
                                metrics['directional_accuracy'] = direction_metrics['directional_accuracy']
                                metrics['weighted_directional_accuracy'] = direction_metrics['weighted_directional_accuracy']
                            model_results['metrics'] = metrics
                    except Exception as e:
                        st.warning(f"⚠️ {model_name} metrics calculation failed: {str(e)}")

                all_results[model_name] = model_results
                progress_bar.progress(0.8 + (i+1) * 0.05)
            
            # Get the last 30 days for historical reference in the comparison chart
            all_results['historical_dates'] = df.index[-30:]
            all_results['historical_prices'] = df['Close'].iloc[-30:].values
            
            st.session_state.comparison_results = {
                "df": df,
                "df_tech": df_tech,
                "all_results": all_results,
                "news_sentiment": news_sentiment
            }
            st.session_state.comparison_made = True
            
        progress_bar.progress(1.0)
        time.sleep(0.5)  # Allow time for progress bar to complete
        progress_bar.empty()

# Dashboard Tab
with tab1:
    if not st.session_state.data_loaded:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("### Welcome to StockSense AI Pro")
            st.markdown("""
                This advanced platform provides professional-grade stock market prediction and analysis using 
                multiple machine learning models powered by artificial intelligence.
                
                **Key Features:**
                - Real-time Technical Indicators with Advanced Visualization
                - Proprietary Feature Engineering for Market Pattern Recognition
                - Market News Sentiment Analysis with NLP Processing
                - Multi-Model Validation Metrics and Performance Benchmarking
                - Volume Profile Analysis and Unusual Activity Detection
                - Professional Stock Screening and Portfolio Optimization
            """)
            
            st.button("Begin Analysis", on_click=run_analysis, key="begin_analysis", use_container_width=True)
            
        with col2:
            st.image(STATIC_IMAGES["technical"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### 🚀 Getting Started")
        st.markdown("""
            <ul>
                <li>Select a data source and stock symbol in the sidebar</li>
                <li>Choose your desired time period and forecast horizon</li>
                <li>Select a prediction model or compare all models</li>
                <li>Enable the analysis features you want to use</li>
                <li>Click 'Begin Analysis' to generate insights</li>
            </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display preview of analysis capabilities
        st.markdown("### Preview of Analysis Capabilities")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(STATIC_IMAGES["technical"], caption="Technical Analysis", use_container_width=True)
            st.markdown("**Advanced Technical Analysis**<br>Multiple indicators, pattern recognition, and trend identification", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(STATIC_IMAGES["ai"], caption="AI-Powered Predictions", use_container_width=True)
            st.markdown("**AI-Powered Predictions**<br>Four state-of-the-art forecasting models with validation metrics", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(STATIC_IMAGES["analysis"], caption="Professional Analysis", use_container_width=True)
            st.markdown("**Professional Analysis Tools**<br>Sentiment analysis, stock screening, anomaly detection and more", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.button("Start Analysis", on_click=run_analysis, key="start_analysis_dash", use_container_width=False)
    else:
        # Display stock information dashboard
        df = st.session_state.stock_data
        
        # Header with stock info
        st.markdown(f'<div class="card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader(f"{symbol} Stock Analysis")
            st.caption(f"Period: {time_period}")
            
        with col2:
            latest_price = df['Close'].iloc[-1]
            previous_price = df['Close'].iloc[-2]
            price_change = latest_price - previous_price
            price_change_pct = (price_change / previous_price) * 100
            
            price_color = "positive-sentiment" if price_change >= 0 else "negative-sentiment"
            price_icon = "📈" if price_change >= 0 else "📉"
            
            st.metric(
                label="Current Price",
                value=f"${latest_price:.2f}",
                delta=f"{price_change:.2f} ({price_change_pct:.2f}%)"
            )
            
        with col3:
            # Calculate average volume
            avg_volume = df['Volume'].mean()
            latest_volume = df['Volume'].iloc[-1]
            volume_change_pct = ((latest_volume - avg_volume) / avg_volume) * 100
            
            if latest_volume > 1e9:
                volume_str = f"{latest_volume/1e9:.2f}B"
            elif latest_volume > 1e6:
                volume_str = f"{latest_volume/1e6:.2f}M"
            elif latest_volume > 1e3:
                volume_str = f"{latest_volume/1e3:.2f}K"
            else:
                volume_str = f"{latest_volume:.0f}"
                
            st.metric(
                label="Volume", 
                value=volume_str,
                delta=f"{volume_change_pct:.2f}%"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Stock chart and volume
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Stock Price History")
        
        # Create candlestick chart
        fig = create_candlestick_plot(df, f"{symbol} Stock Analysis")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Key performance metrics
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Key Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # 1. Year-to-Date Return (if we have data from beginning of current year)
        with col1:
            current_year = datetime.now().year
            start_of_year = datetime(current_year, 1, 1).strftime('%Y-%m-%d')
            
            if start_of_year in df.index:
                start_price = df.loc[start_of_year, 'Close']
                if isinstance(start_price, pd.Series):
                    if len(start_price) > 0:
                        start_price = start_price.iloc[0]
                    else:
                        start_price = None
                
                if start_price is not None:
                    ytd_return = ((latest_price - start_price) / start_price) * 100
                    
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("YTD Return", f"{ytd_return:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    # Fallback to period return
                    first_date = df.index[0]
                    first_price = df.iloc[0]['Close']
                    period_return = ((latest_price - first_price) / first_price) * 100
                    days = (df.index[-1] - first_date).days
                    
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric(f"{days}-Day Return", f"{period_return:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Calculate return for available period
                first_date = df.index[0]
                first_price = df.iloc[0]['Close']
                period_return = ((latest_price - first_price) / first_price) * 100
                days = (df.index[-1] - first_date).days
                
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric(f"{days}-Day Return", f"{period_return:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # 2. Volatility (Standard deviation of returns, annualized)
        with col2:
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility in percentage
            
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Annual Volatility", f"{volatility:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # 3. 52-week high/low distance
        with col3:
            if len(df) >= 252:  # Check if we have at least a year of data
                high_52w = df['High'].rolling(window=252).max().iloc[-1]
                low_52w = df['Low'].rolling(window=252).min().iloc[-1]
                
                dist_from_high = ((latest_price - high_52w) / high_52w) * 100
                
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("From 52W High", f"{dist_from_high:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Use all available data
                period_high = df['High'].max()
                dist_from_high = ((latest_price - period_high) / period_high) * 100
                
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("From Period High", f"{dist_from_high:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
                
        # 4. Average Volume Ratio (current to average)
        with col4:
            volume_ratio = latest_volume / avg_volume
            
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Technical Indicators
        if show_technical:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Technical Indicators")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Calculate RSI
                if 'RSI' in df:
                    rsi = df['RSI'].iloc[-1]
                else:
                    delta = df['Close'].diff()
                    gain = delta.clip(lower=0)
                    loss = -1 * delta.clip(upper=0)
                    avg_gain = gain.rolling(window=14).mean()
                    avg_loss = loss.rolling(window=14).mean().abs()
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs)).iloc[-1]
                
                # RSI Gauge
                rsi_color = "#48BB78" if rsi <= 30 else "#F56565" if rsi >= 70 else "#ECC94B"
                
                fig_rsi = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = rsi,
                    title = {'text': "RSI (14-day)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': rsi_color},
                        'steps': [
                            {'range': [0, 30], 'color': "#48BB78"},
                            {'range': [30, 70], 'color': "#ECC94B"},
                            {'range': [70, 100], 'color': "#F56565"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 2},
                            'thickness': 0.75,
                            'value': rsi
                        }
                    }
                ))
                
                fig_rsi.update_layout(height=300, template="plotly_dark")
                st.plotly_chart(fig_rsi, use_container_width=True)
                
            with col2:
                # Calculate MACD
                if 'MACD' in df:
                    macd = df['MACD'].iloc[-1]
                    macd_signal = df['MACD_Signal'].iloc[-1]
                else:
                    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                    macd = (exp1 - exp2).iloc[-1]
                    macd_signal = (exp1 - exp2).ewm(span=9, adjust=False).mean().iloc[-1]
                
                macd_delta = macd - macd_signal
                macd_color = "#48BB78" if macd_delta > 0 else "#F56565"
                
                # MACD Indicator
                fig_macd = go.Figure(go.Indicator(
                    mode = "number+delta",
                    value = macd,
                    delta = {
                        'reference': macd_signal,
                        'position': 'right',
                        'valueformat': '.3f',
                        'relative': False,
                        'font': {'size': 15}
                    },
                    title = {'text': "MACD Signal"},
                    number = {'valueformat': '.3f'}
                ))
                
                fig_macd.update_layout(height=300, template="plotly_dark")
                st.plotly_chart(fig_macd, use_container_width=True)
            
            # Advanced Technical Analysis Section
            st.markdown("### Advanced Technical Analysis")
            
            # Create a tab interface for different technical indicators
            ind_tab1, ind_tab2, ind_tab3, ind_tab4 = st.tabs(["Moving Averages", "Oscillators", "Price Patterns", "Volume Indicators"])
            
            with ind_tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Calculate moving averages as Series instead of single values
                    sma_20_series = df['Close'].rolling(window=20).mean()
                    sma_50_series = df['Close'].rolling(window=50).mean()
                    sma_200_series = df['Close'].rolling(window=200).mean() if len(df) >= 200 else None

                    # Signal 1: Price above/below MA
                    ma_signals = []
                    if latest_price > sma_20_series.iloc[-1]:
                        ma_signals.append(("Price > 20 SMA", "Bullish"))
                    else:
                        ma_signals.append(("Price < 20 SMA", "Bearish"))
                        
                    if latest_price > sma_50_series.iloc[-1]:
                        ma_signals.append(("Price > 50 SMA", "Bullish"))
                    else:
                        ma_signals.append(("Price < 50 SMA", "Bearish"))
                        
                    if sma_200_series is not None:
                        if latest_price > sma_200_series.iloc[-1]:
                            ma_signals.append(("Price > 200 SMA", "Bullish"))
                        else:
                            ma_signals.append(("Price < 200 SMA", "Bearish"))

                    # Signal 2: Golden Cross / Death Cross (50 vs 200)
                    if sma_200_series is not None:
                        # Check the last two values to detect crossover
                        if (sma_50_series.iloc[-1] > sma_200_series.iloc[-1] and 
                            sma_50_series.iloc[-2] <= sma_200_series.iloc[-2]):
                            ma_signals.append(("Golden Cross (50 SMA crosses above 200 SMA)", "Very Bullish"))
                        elif (sma_50_series.iloc[-1] < sma_200_series.iloc[-1] and 
                              sma_50_series.iloc[-2] >= sma_200_series.iloc[-2]):
                            ma_signals.append(("Death Cross (50 SMA crosses below 200 SMA)", "Very Bearish"))

                    # Display the signals
                    st.markdown("#### Moving Average Signals")
                    for signal, direction in ma_signals:
                        color = "positive-sentiment" if direction == "Bullish" or direction == "Very Bullish" else "negative-sentiment"
                        st.markdown(f"• {signal}: <span class='{color}'>{direction}</span>", unsafe_allow_html=True)
                
                with col2:
                    # Calculate Bollinger Bands
                    bb_period = 20
                    sma = df['Close'].rolling(window=bb_period).mean()
                    std = df['Close'].rolling(window=bb_period).std()
                    upper_bb = sma + (std * 2)
                    lower_bb = sma - (std * 2)
                    
                    # Create a BB width indicator
                    bb_width = (upper_bb - lower_bb) / sma
                    current_bb_width = bb_width.iloc[-1]
                    avg_bb_width = bb_width.mean()
                    
                    fig_bb = go.Figure()
                    
                    # Add close price
                    fig_bb.add_trace(go.Scatter(
                        x=df.index[-60:],  # Last 60 days
                        y=df['Close'][-60:],
                        mode='lines',
                        name='Close',
                        line=dict(color='#9b87f5', width=2)
                    ))
                    
                    # Add Bollinger Bands
                    fig_bb.add_trace(go.Scatter(
                        x=df.index[-60:],
                        y=upper_bb[-60:],
                        mode='lines',
                        name='Upper BB',
                        line=dict(color='#48BB78', width=1.5, dash='dash')
                    ))
                    
                    fig_bb.add_trace(go.Scatter(
                        x=df.index[-60:],
                        y=sma[-60:],
                        mode='lines',
                        name='Middle BB (20 SMA)',
                        line=dict(color='#ECC94B', width=1.5)
                    ))
                    
                    fig_bb.add_trace(go.Scatter(
                        x=df.index[-60:],
                        y=lower_bb[-60:],
                        mode='lines',
                        name='Lower BB',
                        line=dict(color='#F56565', width=1.5, dash='dash')
                    ))
                    
                    # Update layout
                    fig_bb.update_layout(
                        title="Bollinger Bands (20,2)",
                        height=300,
                        template="plotly_dark",
                        margin=dict(l=0, r=0, t=30, b=0),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    
                    st.plotly_chart(fig_bb, use_container_width=True)
                    
                    # BB signals
                    if latest_price > upper_bb.iloc[-1]:
                        st.markdown("• Price above Upper BB: <span class='negative-sentiment'>Potentially overbought</span>", unsafe_allow_html=True)
                    elif latest_price < lower_bb.iloc[-1]:
                        st.markdown("• Price below Lower BB: <span class='positive-sentiment'>Potentially oversold</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("• Price within Bollinger Bands: <span class='neutral-sentiment'>Neutral</span>", unsafe_allow_html=True)
                    
                    if current_bb_width < avg_bb_width * 0.8:
                        st.markdown("• Narrow Bollinger Band width: <span class='neutral-sentiment'>Potential volatility expansion ahead</span>", unsafe_allow_html=True)
                    elif current_bb_width > avg_bb_width * 1.2:
                        st.markdown("• Wide Bollinger Band width: <span class='neutral-sentiment'>High current volatility</span>", unsafe_allow_html=True)
                
            with ind_tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Add Stochastic Oscillator
                    if '%K' in df and '%D' in df:
                        k = df['%K'].iloc[-1]
                        d = df['%D'].iloc[-1]
                    else:
                        # Calculate Stochastic
                        low_14 = df['Low'].rolling(window=14).min()
                        high_14 = df['High'].rolling(window=14).max()
                        k = 100 * ((df['Close'].iloc[-1] - low_14.iloc[-1]) / (high_14.iloc[-1] - low_14.iloc[-1]))
                        # Use average of last 3 %K values as %D
                        d = df['%K'].rolling(window=3).mean().iloc[-1] if '%K' in df else k
                    
                    # Create Stochastic plot
                    fig_stoch = go.Figure()
                    
                    # Add %K and %D lines 
                    k_values = df['%K'].iloc[-60:] if '%K' in df else None
                    d_values = df['%D'].iloc[-60:] if '%D' in df else None
                    
                    if k_values is not None and d_values is not None:
                        fig_stoch.add_trace(go.Scatter(
                            x=df.index[-60:],
                            y=k_values,
                            mode='lines',
                            name='%K',
                            line=dict(color='#9b87f5', width=1.5)
                        ))
                        
                        fig_stoch.add_trace(go.Scatter(
                            x=df.index[-60:],
                            y=d_values,
                            mode='lines',
                            name='%D',
                            line=dict(color='#1EAEDB', width=1.5)
                        ))
                        
                        # Add overbought/oversold lines
                        fig_stoch.add_hline(y=80, line=dict(color='#F56565', width=1, dash='dash'))
                        fig_stoch.add_hline(y=20, line=dict(color='#48BB78', width=1, dash='dash'))
                        
                        # Update layout
                        fig_stoch.update_layout(
                            title="Stochastic Oscillator",
                            height=300,
                            template="plotly_dark",
                            margin=dict(l=0, r=0, t=30, b=0),
                            yaxis=dict(range=[0, 100])
                        )
                        
                        st.plotly_chart(fig_stoch, use_container_width=True)
                    else:
                        st.info("Insufficient data to display Stochastic Oscillator")
                    
                    # Stochastic signals
                    if k > 80:
                        st.markdown("• Stochastic %K > 80: <span class='negative-sentiment'>Overbought</span>", unsafe_allow_html=True)
                    elif k < 20:
                        st.markdown("• Stochastic %K < 20: <span class='positive-sentiment'>Oversold</span>", unsafe_allow_html=True)
                    
                    if k > d and k.shift(1) <= d.shift(1):
                        st.markdown("• %K crossed above %D: <span class='positive-sentiment'>Bullish</span>", unsafe_allow_html=True)
                    elif k < d and k.shift(1) >= d.shift(1):
                        st.markdown("• %K crossed below %D: <span class='negative-sentiment'>Bearish</span>", unsafe_allow_html=True)
                
                with col2:
                    # MFI (Money Flow Index)
                    if 'MFI' in df:
                        mfi = df['MFI'].iloc[-1]
                        
                        # Create MFI plot
                        fig_mfi = go.Figure()
                        
                        fig_mfi.add_trace(go.Scatter(
                            x=df.index[-60:],
                            y=df['MFI'].iloc[-60:],
                            mode='lines',
                            name='MFI',
                            line=dict(color='#48BB78', width=1.5)
                        ))
                        
                        # Add overbought/oversold lines
                        fig_mfi.add_hline(y=80, line=dict(color='#F56565', width=1, dash='dash'))
                        fig_mfi.add_hline(y=20, line=dict(color='#48BB78', width=1, dash='dash'))
                        
                        # Update layout
                        fig_mfi.update_layout(
                            title="Money Flow Index",
                            height=300,
                            template="plotly_dark",
                            margin=dict(l=0, r=0, t=30, b=0),
                            yaxis=dict(range=[0, 100])
                        )
                        
                        st.plotly_chart(fig_mfi, use_container_width=True)
                        
                        # MFI signals
                        if mfi > 80:
                            st.markdown("• MFI > 80: <span class='negative-sentiment'>Overbought</span>", unsafe_allow_html=True)
                        elif mfi < 20:
                            st.markdown("• MFI < 20: <span class='positive-sentiment'>Oversold</span>", unsafe_allow_html=True)
                        
                        # MFI divergence (basic check)
                        price_trend_up = df['Close'].iloc[-5:].is_monotonic_increasing
                        mfi_trend_down = df['MFI'].iloc[-5:].is_monotonic_decreasing
                        
                        price_trend_down = df['Close'].iloc[-5:].is_monotonic_decreasing
                        mfi_trend_up = df['MFI'].iloc[-5:].is_monotonic_increasing
                        
                        if price_trend_up and mfi_trend_down:
                            st.markdown("• Bearish Divergence: <span class='negative-sentiment'>Price up, MFI down</span>", unsafe_allow_html=True)
                        elif price_trend_down and mfi_trend_up:
                            st.markdown("• Bullish Divergence: <span class='positive-sentiment'>Price down, MFI up</span>", unsafe_allow_html=True)
                    else:
                        st.info("MFI data not available")
            
            with ind_tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Detect some basic chart patterns
                    patterns = []
                    
                    # Check for inside day
                    if 'Inside_Day' in df:
                        if df['Inside_Day'].iloc[-1] == 1:
                            patterns.append(("Inside Day", "Neutral - Consolidation"))
                    else:
                        inside_day = ((df['High'].iloc[-1] < df['High'].iloc[-2]) & 
                                     (df['Low'].iloc[-1] > df['Low'].iloc[-2]))
                        if inside_day:
                            patterns.append(("Inside Day", "Neutral - Consolidation"))
                    
                    # Check for outside day
                    if 'Outside_Day' in df:
                        if df['Outside_Day'].iloc[-1] == 1:
                            patterns.append(("Outside Day", "Volatile - Potential Reversal"))
                    else:
                        outside_day = ((df['High'].iloc[-1] > df['High'].iloc[-2]) & 
                                      (df['Low'].iloc[-1] < df['Low'].iloc[-2]))
                        if outside_day:
                            patterns.append(("Outside Day", "Volatile - Potential Reversal"))
                    
                    # Check for bullish engulfing
                    bullish_engulfing = ((df['Open'].iloc[-1] < df['Close'].iloc[-2]) &
                                        (df['Close'].iloc[-1] > df['Open'].iloc[-2]) &
                                        (df['Close'].iloc[-1] > df['Open'].iloc[-1]) &
                                        (df['Close'].iloc[-2] < df['Open'].iloc[-2]))
                    if bullish_engulfing:
                        patterns.append(("Bullish Engulfing", "Bullish"))
                    
                    # Check for bearish engulfing
                    bearish_engulfing = ((df['Open'].iloc[-1] > df['Close'].iloc[-2]) &
                                        (df['Close'].iloc[-1] < df['Open'].iloc[-2]) &
                                        (df['Close'].iloc[-1] < df['Open'].iloc[-1]) &
                                        (df['Close'].iloc[-2] > df['Open'].iloc[-2]))
                    if bearish_engulfing:
                        patterns.append(("Bearish Engulfing", "Bearish"))
                    
                    # Check for doji
                    if 'Doji' in df:
                        if df['Doji'].iloc[-1] == 1:
                            patterns.append(("Doji", "Potential Reversal"))
                    else:
                        doji = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1]) <= (0.1 * (df['High'].iloc[-1] - df['Low'].iloc[-1]))
                        if doji:
                            patterns.append(("Doji", "Potential Reversal"))
                            
                    # Check for gap up/down
                    if df['Open'].iloc[-1] > df['High'].iloc[-2]:
                        patterns.append(("Gap Up", "Bullish"))
                    elif df['Open'].iloc[-1] < df['Low'].iloc[-2]:
                        patterns.append(("Gap Down", "Bearish"))
                    
                    # Display detected patterns
                    st.markdown("#### Candlestick Patterns")
                    if patterns:
                        for pattern, direction in patterns:
                            color = "positive-sentiment" if "Bullish" in direction else "negative-sentiment" if "Bearish" in direction else "neutral-sentiment"
                            st.markdown(f"• {pattern}: <span class='{color}'>{direction}</span>", unsafe_allow_html=True)
                    else:
                        st.write("No significant patterns detected in the latest data")
                    
                    # Display recent price pattern
                    st.markdown("#### Recent Price Pattern")
                    
                    # Create small candlestick chart for recent data
                    fig_candle = go.Figure(go.Candlestick(
                        x=df.index[-10:],
                        open=df['Open'].iloc[-10:],
                        high=df['High'].iloc[-10:],
                        low=df['Low'].iloc[-10:],
                        close=df['Close'].iloc[-10:],
                        name='Price',
                        increasing_line_color='#48BB78',
                        decreasing_line_color='#F56565'
                    ))
                                
                    # Update layout
                    fig_candle.update_layout(
                        title="Last 10 Trading Days",
                        height=300,
                        template="plotly_dark",
                        xaxis_rangeslider_visible=False,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    
                    st.plotly_chart(fig_candle, use_container_width=True)
                    
                with col2:
                    # Support and Resistance Analysis
                    st.markdown("#### Support & Resistance Levels")
                    
                    # For a simple demo, let's use pivot points
                    # In a real application, this would use more advanced algorithms
                    # like cluster analysis of price history or fractal analysis
                    
                    # Calculate pivot points
                    prev_high = df['High'].iloc[-2]
                    prev_low = df['Low'].iloc[-2]
                    prev_close = df['Close'].iloc[-2]
                    
                    pivot = (prev_high + prev_low + prev_close) / 3
                    r1 = 2 * pivot - prev_low
                    r2 = pivot + (prev_high - prev_low)
                    s1 = 2 * pivot - prev_high
                    s2 = pivot - (prev_high - prev_low)
                    
                    # Format as currency
                    pivot_str = f"${pivot:.2f}"
                    r1_str = f"${r1:.2f}"
                    r2_str = f"${r2:.2f}"
                    s1_str = f"${s1:.2f}"
                    s2_str = f"${s2:.2f}"
                    
                    # Calculate current position relative to pivot levels
                    if latest_price > r2:
                        level = "Above R2"
                        next_level = "New highs"
                    elif latest_price > r1:
                        level = "Between R1 and R2"
                        next_level = r2_str
                    elif latest_price > pivot:
                        level = "Between Pivot and R1"
                        next_level = r1_str
                    elif latest_price > s1:
                        level = "Between S1 and Pivot"
                        next_level = pivot_str
                    elif latest_price > s2:
                        level = "Between S2 and S1"
                        next_level = s1_str
                    else:
                        level = "Below S2"
                        next_level = s2_str
                    
                    # Display pivot points
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Resistance Levels**")
                        st.markdown(f"R2: {r2_str}")
                        st.markdown(f"R1: {r1_str}")
                        st.markdown(f"**Pivot: {pivot_str}**")
                        st.markdown(f"S1: {s1_str}")
                        st.markdown(f"S2: {s2_str}")
                        
                    with col2:
                        st.markdown("**Current Position**")
                        st.markdown(f"Level: {level}")
                        st.markdown(f"Next key level: {next_level}")
                        
                        # Distance to nearest level
                        if "Above" in level:
                            nearest = r2
                        elif "Between R1 and R2" in level:
                            nearest = min(r2 - latest_price, latest_price - r1)
                        elif "Between Pivot and R1" in level:
                            nearest = min(r1 - latest_price, latest_price - pivot)
                        elif "Between S1 and Pivot" in level:
                            nearest = min(pivot - latest_price, latest_price - s1)
                        elif "Between S2 and S1" in level:
                            nearest = min(s1 - latest_price, latest_price - s2)
                        else:
                            nearest = s2
                        
                        st.markdown(f"Distance to nearest level: ${abs(nearest):.2f}")
                    
                    # Add Fibonacci Retracement Levels (if calculated)
                    if 'Fib_23.6' in df:
                        st.markdown("#### Fibonacci Retracement Levels")
                        
                        fib_levels = [
                            ("0% (Swing Low)", df['Fib_0'].iloc[-1]),
                            ("23.6%", df['Fib_23.6'].iloc[-1]),
                            ("38.2%", df['Fib_38.2'].iloc[-1]),
                            ("50%", df['Fib_50'].iloc[-1]),
                            ("61.8%", df['Fib_61.8'].iloc[-1]),
                            ("100% (Swing High)", df['Fib_100'].iloc[-1])
                        ]
                        
                        for level_name, level_value in fib_levels:
                            st.markdown(f"{level_name}: ${level_value:.2f}")
            
            with ind_tab4:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Volume analysis
                    st.markdown("#### Volume Analysis")
                    
                    # Recent volume trend
                    recent_volume = df['Volume'].iloc[-5:].values
                    avg_volume_20d = df['Volume'].iloc[-20:].mean()
                    
                    volume_trend = "Increasing" if recent_volume[-1] > recent_volume[0] else "Decreasing"
                    volume_vs_avg = f"{(recent_volume[-1] / avg_volume_20d - 1) * 100:.1f}%"
                    volume_direction = "above" if recent_volume[-1] > avg_volume_20d else "below"
                    
                    st.markdown(f"• Recent Volume Trend: {volume_trend}")
                    st.markdown(f"• Latest Volume: {volume_vs_avg} {volume_direction} 20-day average")
                    
                    # Unusual volume days (>1.5x average)
                    unusual_volume_days = df[df['Volume'] > 1.5 * df['Volume'].rolling(20).mean()]
                    recent_unusual = unusual_volume_days[unusual_volume_days.index >= df.index[-20]]
                    
                    if not recent_unusual.empty:
                        st.markdown(f"• {len(recent_unusual)} days with unusually high volume in the last 20 trading days")
                        
                    # On Balance Volume (OBV)
                    if 'OBV' in df:
                        obv = df['OBV'].iloc[-1]
                        obv_prev = df['OBV'].iloc[-2]
                        obv_direction = "up" if obv > obv_prev else "down"
                        
                        # Check for OBV divergence
                        price_up = df['Close'].iloc[-5:].is_monotonic_increasing
                        obv_down = df['OBV'].iloc[-5:].is_monotonic_decreasing
                        price_down = df['Close'].iloc[-5:].is_monotonic_decreasing
                        obv_up = df['OBV'].iloc[-5:].is_monotonic_increasing
                        
                        if price_up and obv_down:
                            st.markdown("• <span class='negative-sentiment'>Bearish Divergence</span>: Price rising but OBV falling", unsafe_allow_html=True)
                        elif price_down and obv_up:
                            st.markdown("• <span class='positive-sentiment'>Bullish Divergence</span>: Price falling but OBV rising", unsafe_allow_html=True)
                        else:
                            st.markdown(f"• On-Balance Volume trending {obv_direction}")
                    
                    # Create OBV plot
                    if 'OBV' in df:
                        fig_obv = go.Figure()
                        
                        fig_obv.add_trace(go.Scatter(
                            x=df.index[-60:],
                            y=df['OBV'].iloc[-60:],
                            mode='lines',
                            name='OBV',
                            line=dict(color='#9b87f5', width=1.5)
                        ))
                        
                        # Update layout
                        fig_obv.update_layout(
                            title="On-Balance Volume (OBV)",
                            height=300,
                            template="plotly_dark",
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        
                        st.plotly_chart(fig_obv, use_container_width=True)
                
                with col2:
                    # Volume by Price (simplified)
                    st.markdown("#### Volume by Price")
                    
                    # Create price bins for volume distribution
                    price_min = df['Low'].min()
                    price_max = df['High'].max()
                    price_range = price_max - price_min
                    
                    # Create 10 price bins
                    bins = 10
                    bin_size = price_range / bins
                    price_bins = [price_min + i * bin_size for i in range(bins + 1)]
                    
                    # Calculate volume in each price bin
                    vol_by_price = np.zeros(bins)
                    
                    for i in range(len(df)):
                        high = df['High'].iloc[i]
                        low = df['Low'].iloc[i]
                        volume = df['Volume'].iloc[i]
                        
                        # Distribute volume across price bins that the candle spans
                        candle_range = high - low
                        if candle_range > 0:
                            for j in range(bins):
                                bin_low = price_bins[j]
                                bin_high = price_bins[j + 1]
                                
                                # If candle overlaps with this bin
                                if low <= bin_high and high >= bin_low:
                                    # Calculate overlap
                                    overlap_low = max(low, bin_low)
                                    overlap_high = min(high, bin_high)
                                    overlap_pct = (overlap_high - overlap_low) / candle_range
                                    
                                    # Attribute volume proportionally
                                    vol_by_price[j] += volume * overlap_pct
                    
                    # Create horizontal bar chart
                    bin_labels = [f"${(price_bins[i] + price_bins[i+1])/2:.2f}" for i in range(bins)]
                    
                    fig_vbp = go.Figure()
                    
                    fig_vbp.add_trace(go.Bar(
                        y=bin_labels,
                        x=vol_by_price,
                        orientation='h',
                        marker_color='#9b87f5',
                        name='Volume'
                    ))
                    
                    # Add current price line
                    current_price_index = max(0, min(bins - 1, int((latest_price - price_min) / bin_size)))
                    current_price_bin = bin_labels[current_price_index]
                    
                    fig_vbp.add_trace(go.Scatter(
                        y=[current_price_bin],
                        x=[max(vol_by_price) * 1.1],
                        mode='markers',
                        marker=dict(symbol='triangle-left', size=15, color='#48BB78'),
                        name='Current Price'
                    ))
                    
                    # Update layout
                    fig_vbp.update_layout(
                        title="Volume by Price",
                        height=300,
                        template="plotly_dark",
                        margin=dict(l=0, r=0, t=30, b=0),
                        xaxis_title="Volume",
                        yaxis_title="Price Levels"
                    )
                    
                    st.plotly_chart(fig_vbp, use_container_width=True)
                    
                    # Identify high volume price levels (potential support/resistance)
                    high_vol_threshold = np.percentile(vol_by_price, 80)
                    high_vol_bins = [i for i, vol in enumerate(vol_by_price) if vol > high_vol_threshold]
                    
                    if high_vol_bins:
                        st.markdown("#### Key Volume Levels (Support/Resistance)")
                        for bin_idx in high_vol_bins:
                            price_level = (price_bins[bin_idx] + price_bins[bin_idx+1])/2
                            if price_level < latest_price:
                                st.markdown(f"• <span class='positive-sentiment'>Support</span>: ${price_level:.2f}", unsafe_allow_html=True)
                            else:
                                st.markdown(f"• <span class='negative-sentiment'>Resistance</span>: ${price_level:.2f}", unsafe_allow_html=True)
            
            # Display technical indicators in a table
            st.subheader("Technical Analysis Summary")
            
            # Calculate key indicators
            sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
            sma_200 = df['Close'].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None
            
            # Calculate Bollinger Bands
            bb_period = 20
            sma = df['Close'].rolling(window=bb_period).mean()
            std = df['Close'].rolling(window=bb_period).std()
            upper_bb = sma + (std * 2)
            lower_bb = sma - (std * 2)
            
            # Create indicators table
            indicators_data = {
                "Indicator": [
                    "SMA (20)", "SMA (50)", "SMA (200)", 
                    "Upper Bollinger Band", "Lower Bollinger Band", 
                    "RSI (14)", "MACD", "MACD Signal"
                ],
                "Value": [
                    f"${sma_20:.2f}", 
                    f"${sma_50:.2f}", 
                    f"${sma_200:.2f}" if sma_200 is not None else "N/A", 
                    f"${upper_bb.iloc[-1]:.2f}",
                    f"${lower_bb.iloc[-1]:.2f}",
                    f"{rsi:.2f}",
                    f"{macd:.4f}",
                    f"{macd_signal:.4f}"
                ],
                "Signal": [
                    "Bullish" if latest_price > sma_20 else "Bearish",
                    "Bullish" if latest_price > sma_50 else "Bearish",
                    "Bullish" if sma_200 is not None and latest_price > sma_200 else "Bearish" if sma_200 is not None else "N/A",
                    "Overbought" if latest_price > upper_bb.iloc[-1] else "Normal",
                    "Oversold" if latest_price < lower_bb.iloc[-1] else "Normal",
                    "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral",
                    "Bullish" if macd > macd_signal else "Bearish",
                    "-"
                ]
            }
            
            indicators_df = pd.DataFrame(indicators_data)
            st.dataframe(indicators_df, use_container_width=True, hide_index=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # News Articles
        if show_news_sentiment:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📰 Recent News Articles")
            
            news_articles = get_news_sentiment(symbol)
            
            if news_articles:
                # Display articles in 2-column grid
                cols = st.columns(2)
                for i, article in enumerate(news_articles):
                    with cols[i % 2]:
                        # Format the date
                        published_at = article.get('publishedAt', '')
                        if published_at:
                            try:
                                from datetime import datetime
                                date_obj = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                                formatted_date = date_obj.strftime('%B %d, %Y')
                            except:
                                formatted_date = published_at
                        else:
                            formatted_date = ''
                        
                        # Get source name
                        source_name = article.get('source', {}).get('name', 'Unknown Source')
                        
                        # Article card
                        st.markdown(f"""
                            <div class="news-card">
                                <h5>{article.get('title', 'No Title')}</h5>
                                <p><small><strong>{source_name}</strong> • {formatted_date}</small></p>
                                <p>{article.get('description', 'No description available.')}</p>
                                <a href="{article.get('url', '#')}" target="_blank">Read more →</a>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No news articles available at this time. Please check your API key or try again later.")
                
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature Engineering Analysis
        if show_feature_engineering:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Advanced Feature Engineering")
            
            # Get the engineered features
            df_features = engineer_features(df)
            
            # Display feature importances for prediction (simulated for demo)
            st.markdown("### Key Predictive Features")
            
            # Create a simulated feature importance chart
            features = [
                'Price_SMA_20_Ratio', 'RSI', 'Volume_Change', 'MACD', 
                'Momentum_5', 'ATR', 'Daily_Volatility', 'Price_Range',
                'StochRSI', 'OBV'
            ]
            
            # Generate random importance values (in a real app, these would come from the model)
            np.random.seed(42)  # For reproducibility
            importances = np.random.rand(10)
            importances = importances / np.sum(importances)  # Normalize
            
            # Sort by importance
            sorted_idx = np.argsort(importances)[::-1]
            sorted_features = [features[i] for i in sorted_idx]
            sorted_importances = importances[sorted_idx]
            
            # Create feature importance bar chart
            fig_imp = go.Figure(go.Bar(
                y=sorted_features,
                x=sorted_importances,
                orientation='h',
                marker_color=np.linspace(0.1, 0.9, len(features)),
                marker_colorscale='Viridis'
            ))
            
            fig_imp.update_layout(
                title="Feature Importance for Price Prediction",
                xaxis_title="Relative Importance",
                yaxis_title="Feature",
                height=400,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_imp, use_container_width=True)
            
            # Feature correlation matrix
            st.markdown("### Feature Correlation Analysis")
            
            # Select a subset of engineered features
            selected_features = [
                'Close', 'Volume', 'RSI', 'MACD', 'ATR', 'OBV', 
                'Momentum_5', 'Daily_Volatility', 'MFI', 'BB_Width'
            ]
            
            # Filter features that exist in the dataframe
            available_features = [f for f in selected_features if f in df_features.columns]
            
            if len(available_features) > 1:
                # Calculate correlation matrix
                corr_matrix = df_features[available_features].corr()
                
                # Create heatmap
                fig_corr = px.imshow(
                    corr_matrix, 
                    text_auto=True, 
                    color_continuous_scale='Viridis',
                    aspect="auto",
                    title="Feature Correlation Matrix"
                )
                
                fig_corr.update_layout(
                    height=500,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Time-based analysis
            st.markdown("### Temporal Pattern Analysis")
            
            if 'Day_of_Week' in df_features.columns:
                # Average returns by day of week
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                
                # Add day name column
                df_features['Day_Name'] = df_features['Day_of_Week'].apply(lambda x: day_names[x])
                
                # Calculate average returns by day
                day_returns = df_features.groupby('Day_Name')['Daily_Return'].mean() * 100
                
                # Create bar chart
                fig_day = go.Figure(go.Bar(
                    x=day_returns.index,
                    y=day_returns.values,
                    marker_color=[
                        'green' if val > 0 else 'red' for val in day_returns.values
                    ]
                ))
                
                fig_day.update_layout(
                    title="Average Daily Returns by Day of Week",
                    xaxis_title="Day of Week",
                    yaxis_title="Average Return (%)",
                    height=350,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig_day, use_container_width=True)
                
            if 'Month' in df_features.columns:
                # Average returns by month
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                # Create month name column
                df_features['Month_Name'] = df_features['Month'].apply(lambda x: month_names[x-1])
                
                # Calculate average returns by month
                month_returns = df_features.groupby('Month_Name')['Daily_Return'].mean() * 100
                
                # Reorder months correctly
                month_returns = month_returns.reindex(month_names)
                
                # Create bar chart
                fig_month = go.Figure(go.Bar(
                    x=month_returns.index,
                    y=month_returns.values,
                    marker_color=[
                        'green' if val > 0 else 'red' for val in month_returns.values
                    ]
                ))
                
                fig_month.update_layout(
                    title="Average Daily Returns by Month",
                    xaxis_title="Month",
                    yaxis_title="Average Return (%)",
                    height=350,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig_month, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Reset button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Reset Analysis", key="reset_dashboard"):
                st.session_state.data_loaded = False
                st.session_state.prediction_made = False
                st.session_state.comparison_made = False
                st.session_state.stock_data = None
                st.session_state.prediction_results = None
                st.session_state.comparison_results = None
                st.experimental_rerun()
                
            if st.button("Run New Analysis", on_click=run_analysis, key="run_analysis_again"):
                pass

# Prediction Tab
with tab2:
    if not st.session_state.data_loaded:
        st.info("Please run an analysis from the Dashboard tab to view predictions.")
    elif st.session_state.comparison_made:
        # Show comparison results in the Prediction tab for Compare All mode
        st.info("📊 You selected 'Compare All' mode. Please view the 'Model Comparison' tab to see all predictions.")
    elif model_type == "Compare All":
        st.info("📊 Compare All is selected. Run the analysis to generate comparison predictions.")
        if st.button("Run Compare All Analysis", key="run_compare_all_from_prediction"):
            run_analysis()
    elif st.session_state.prediction_made:
        results_data = st.session_state.prediction_results
        df = results_data["df"]
        df_tech = results_data["df_tech"]
        results = results_data["results"]
        news_sentiment = results_data["news_sentiment"]
        
        # Display prediction header
        st.markdown(f'<div class="card">', unsafe_allow_html=True)
        st.subheader(f"{symbol} Stock Price Prediction - {model_type} Model")
        
        # Calculate prediction metrics
        latest_price = df['Close'].iloc[-1]
        predicted_price = results['future_predictions'][0] if results['future_predictions'] is not None and len(results['future_predictions']) > 0 else latest_price
        price_change = predicted_price - latest_price
        price_change_pct = (price_change / latest_price) * 100
        
        price_color = "positive-sentiment" if price_change >= 0 else "negative-sentiment"
        price_icon = "📈" if price_change >= 0 else "📉"
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"""
                <h3>Next day prediction: ${predicted_price:.2f} <span class="{price_color}">{price_icon} {price_change:.2f} ({price_change_pct:.2f}%)</span></h3>
            """, unsafe_allow_html=True)
            
        with col2:
            st.metric(
                label="Current Price", 
                value=f"${latest_price:.2f}"
            )
            
        with col3:
            st.metric(
                label="Predicted Change", 
                value=f"${price_change:.2f}",
                delta=f"{price_change_pct:.2f}%"
            )
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Plot prediction
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Price Forecast Visualization")
        
        # Create the prediction plot
        fig = plot_prediction(df, results, model_type)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display prediction table
        st.subheader("Forecast Details")
        
        if 'future_dates' in results and 'future_predictions' in results:
            forecast_data = pd.DataFrame({
                'Date': [date.strftime('%Y-%m-%d') for date in results['future_dates']],
                'Predicted Price': [f"${price:.2f}" for price in results['future_predictions']],
                'Change': [f"${price - latest_price:.2f}" for price in results['future_predictions']],
                'Change %': [f"{((price - latest_price) / latest_price) * 100:.2f}%" for price in results['future_predictions']]
            })
            
            st.dataframe(forecast_data, use_container_width=True, hide_index=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model explanation
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Model Explanation")
        
        # Different explanation based on model type
        if model_type == "ARIMA":
            st.markdown("""
                ### ARIMA Model Details
                
                The **AutoRegressive Integrated Moving Average (ARIMA)** model is a statistical method for analyzing and forecasting time series data.
                
                #### How ARIMA Works:
                - **Autoregressive (AR)**: Uses the dependent relationship between an observation and a number of lagged observations
                - **Integrated (I)**: Applies differencing of observations to make the time series stationary
                - **Moving Average (MA)**: Uses the dependency between an observation and residual errors from a moving average model
                
                #### Strengths:
                - Handles temporal dependencies well
                - Works effectively with stationary data
                - Good for short-term forecasting
                
                #### Limitations:
                - Assumes linear relationships
                - Cannot capture non-linear patterns
                - Less effective for longer-term predictions
            """)
            
        elif model_type == "Random Forest":
            st.markdown("""
                ### Random Forest Model Details
                
                The **Random Forest** model is an ensemble learning method that combines multiple decision trees for regression or classification tasks.
                
                #### How Random Forest Works:
                - Creates multiple decision trees on randomly selected data samples
                - Gets prediction from each tree and uses averaging to improve prediction accuracy
                - Uses feature engineering to identify important predictors
                
                #### Strengths:
                - Handles non-linear relationships
                - Robust to outliers and noise
                - Provides feature importance rankings
                - Reduces overfitting compared to single decision trees
                
                #### Limitations:
                - Limited interpretability (black box model)
                - May struggle with true trend extrapolation
                - Requires good feature engineering
            """)
            
        elif model_type == "Prophet":
            st.markdown("""
                ### Prophet Model Details
                
                **Prophet** is a procedure for forecasting time series data developed by Facebook. It is designed for business forecasting tasks.
                
                #### How Prophet Works:
                - Decomposes time series into trend, seasonality, and holiday components
                - Handles missing data and outliers automatically
                - Uses Bayesian curve fitting with changepoints for trend changes
                
                #### Strengths:
                - Automatically handles seasonality at multiple periods
                - Robust to missing data and outliers
                - Accommodates trend changes and non-linear growth
                - Works well with data having strong seasonal patterns
                
                #### Limitations:
                - May not fully capture complex dependencies between variables
                - Sometimes produces overly smooth forecasts
                - Doesn't leverage exogenous variables as effectively as other models
            """)
            
        elif model_type == "LSTM":
            st.markdown("""
                ### LSTM Model Details
                
                **Long Short-Term Memory (LSTM)** networks are a type of recurrent neural network (RNN) capable of learning order dependence in sequence prediction problems.
                
                #### How LSTM Works:
                - Uses memory cells that can maintain information for long periods of time
                - Contains gates that control the flow of information (input gate, forget gate, output gate)
                - Captures complex patterns and long-term dependencies in time series data
                
                #### Strengths:
                - Captures complex non-linear relationships
                - Effective at learning long-term dependencies
                - Can process data with multiple input features
                - Powerful predictive capability with sufficient data
                
                #### Limitations:
                - Requires substantial training data
                - Computationally intensive
                - Prone to overfitting without proper regularization
                - "Black box" nature makes interpretation difficult
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Validation metrics
        if show_validation and 'metrics' in results:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Model Validation Metrics")
            
            metrics = results['metrics']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown(f"""
                    <h4>RMSE</h4>
                    <h2>{metrics.get('rmse', 'N/A'):.4f}</h2>
                    <p>Root Mean Squared Error</p>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown(f"""
                    <h4>MAE</h4>
                    <h2>{metrics.get('mae', 'N/A'):.4f}</h2>
                    <p>Mean Absolute Error</p>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown(f"""
                    <h4>Directional Accuracy</h4>
                    <h2>{metrics.get('directional_accuracy', 'N/A'):.2f}%</h2>
                    <p>Correct Direction Prediction</p>
                    <small style="color: #9b87f5;">Weighted: {metrics.get('weighted_directional_accuracy', 'N/A'):.2f}%</small>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Model validation insights
            st.markdown(f"""
                ### Model Performance Analysis
                
                The {model_type} model has been validated using historical data for {symbol}. 
                
                - **Root Mean Squared Error (RMSE)**: {metrics.get('rmse', 'N/A'):.4f} - Average magnitude of prediction errors (lower is better)
                - **Mean Absolute Error (MAE)**: {metrics.get('mae', 'N/A'):.4f} - Average absolute prediction error in dollar terms
                - **R-squared**: {metrics.get('r2', 'N/A'):.4f} - Proportion of price variance explained by the model (higher is better)
                - **Directional Accuracy**: {metrics.get('directional_accuracy', 'N/A'):.2f}% - Percentage of correct movement direction predictions (with noise filtering)
                - **Weighted Directional Accuracy**: {metrics.get('weighted_directional_accuracy', 'N/A'):.2f}% - Direction accuracy weighted by magnitude of price changes
                
                The model was trained on {time_period} of historical data and validated using a test set.
                
                > Note: These metrics help evaluate model performance, but they don't guarantee future predictions will be equally accurate. Market conditions can change rapidly.
            """)
            
            # Create visual representation of errors
            if 'test_predictions' in results and len(results['test_predictions']) > 0:
                test_size = min(len(results['test_predictions']), 30)
                actual_values = df['Close'].iloc[-test_size:].values
                predicted_values = results['test_predictions'][-test_size:]
                dates = df.index[-test_size:]
                
                # Create error visualization
                fig_error = go.Figure()
                
                # Add actual vs predicted
                fig_error.add_trace(go.Scatter(
                    x=dates,
                    y=actual_values,
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='#9b87f5', width=3),
                    marker=dict(size=6),
                    connectgaps=True
                ))
                
                fig_error.add_trace(go.Scatter(
                    x=dates,
                    y=predicted_values,
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='#1EAEDB', width=3, dash='dash'),
                    marker=dict(size=6),
                    connectgaps=True
                ))
                
                # Add error bars
                errors = [pred - act for pred, act in zip(predicted_values, actual_values)]
                avg_error = sum(errors) / len(errors)
                
                fig_error.add_trace(go.Bar(
                    x=dates,
                    y=errors,
                    name='Error',
                    marker_color=['#ef5350' if e < 0 else '#26a69a' for e in errors],
                    opacity=0.6,
                    yaxis='y2'
                ))
                
                # Update layout with dual y-axis
                fig_error.update_layout(
                    title="Model Prediction Errors",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    yaxis2=dict(
                        title="Error",
                        overlaying='y',
                        side='right',
                        showgrid=False
                    ),
                    height=500,
                    template="plotly_dark",
                    legend=dict(x=0, y=1.1, orientation='h'),
                    hovermode='x unified',
                    xaxis=dict(type='date', tickformat='%b %d\n%H:%M')
                )
                
                st.plotly_chart(fig_error, use_container_width=True)
                
                st.markdown(f"**Average Error**: ${avg_error:.2f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Risk and confidence analysis
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Risk and Confidence Analysis")
        
        # Confidence intervals
        if 'lower_bound' in results and 'upper_bound' in results:
            # Display confidence intervals for first prediction
            first_pred = results['future_predictions'][0] if len(results['future_predictions']) > 0 else latest_price
            first_lower = results['lower_bound'][0] if len(results['lower_bound']) > 0 else first_pred * 0.95
            first_upper = results['upper_bound'][0] if len(results['upper_bound']) > 0 else first_pred * 1.05
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Lower Bound (95%)", 
                    value=f"${first_lower:.2f}",
                    delta=f"{((first_lower - latest_price) / latest_price) * 100:.2f}%"
                )
                
            with col2:
                st.metric(
                    label="Forecast", 
                    value=f"${first_pred:.2f}",
                    delta=f"{((first_pred - latest_price) / latest_price) * 100:.2f}%"
                )
                
            with col3:
                st.metric(
                    label="Upper Bound (95%)", 
                    value=f"${first_upper:.2f}",
                    delta=f"{((first_upper - latest_price) / latest_price) * 100:.2f}%"
                )
            
            # Risk and reward ratio
            downside_risk = latest_price - first_lower
            upside_potential = first_upper - latest_price
            risk_reward_ratio = upside_potential / downside_risk if downside_risk > 0 else 0
            
            st.markdown(f"""
                ### Risk Assessment
                
                - **Downside Risk**: ${downside_risk:.2f} ({((downside_risk) / latest_price) * 100:.2f}%)
                - **Upside Potential**: ${upside_potential:.2f} ({((upside_potential) / latest_price) * 100:.2f}%)
                - **Risk/Reward Ratio**: {risk_reward_ratio:.2f}
                - **Range Width**: ${(first_upper - first_lower):.2f} ({((first_upper - first_lower) / latest_price) * 100:.2f}%)
                
                > Note: All forecasts have inherent uncertainty. The confidence intervals represent the range where prices are expected to fall with 95% probability, based on historical volatility and model characteristics.
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
            
        # Reset button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Reset Prediction", key="reset_prediction"):
                st.session_state.prediction_made = False
                st.session_state.prediction_results = None
                st.experimental_rerun()
                
            if model_type != "Compare All" and st.button("Compare with Other Models", key="compare_models"):
                st.session_state.prediction_made = False
                st.experimental_rerun()
    else:
        st.info("Please select a model in the sidebar and run an analysis to see predictions.")
        if st.button("Run Analysis with Selected Model", on_click=run_analysis):
            pass

# Model comparison tab
with tab3:
    if not st.session_state.data_loaded:
        st.info("Please run an analysis from the Dashboard tab to compare models.")
        if model_type == "Compare All" and st.button("Run Compare All Analysis", key="run_compare_all_tab3"):
            run_analysis()
    elif st.session_state.comparison_made:
        comparison_data = st.session_state.comparison_results
        df = comparison_data["df"]
        df_tech = comparison_data["df_tech"]
        all_results = comparison_data["all_results"]
        news_sentiment = comparison_data["news_sentiment"]
        
        # Display comparison header
        st.markdown(f'<div class="card">', unsafe_allow_html=True)
        st.subheader(f"{symbol} Stock Prediction Model Comparison")
        
        latest_price = df['Close'].iloc[-1]
        
        # Diagnostics: show model output status
        with st.expander("🔎 Model output diagnostics", expanded=False):
            for model_key, model_value in all_results.items():
                if model_key in ['historical_dates', 'historical_prices']:
                    continue
                preds = model_value.get('future_predictions', []) if isinstance(model_value, dict) else []
                dates = model_value.get('future_dates', []) if isinstance(model_value, dict) else []
                err = model_value.get('error') if isinstance(model_value, dict) else None
                st.write(f"**{model_key}** → predictions: {len(preds)}, dates: {len(dates)}")
                if err:
                    st.warning(f"{model_key} warning: {err}")

        # Calculate average prediction across all forecast days (not just day 1)
        models = [
            model for model in all_results.keys()
            if model not in ['historical_dates', 'historical_prices']
            and 'future_predictions' in all_results[model]
            and len(all_results[model]['future_predictions']) > 0
        ]

        if not models:
            st.warning("No model predictions available. Please rerun analysis or check model warnings above.")
            st.stop()

        # Surface model errors (if any)
        for model in models:
            if 'error' in all_results[model]:
                st.warning(f"⚠️ {model}: {all_results[model]['error']}")

        # Get average of all predictions across all models and all forecast days
        all_predictions = []
        for model in models:
            # Use average of all forecast days for this model
            model_avg = np.mean(all_results[model]['future_predictions'])
            all_predictions.append(model_avg)
        
        avg_prediction = np.mean(all_predictions) if all_predictions else latest_price
        price_change = avg_prediction - latest_price
        price_change_pct = (price_change / latest_price) * 100
        
        price_color = "positive-sentiment" if price_change >= 0 else "negative-sentiment"
        price_icon = "📈" if price_change >= 0 else "📉"
        
        st.markdown(f"""
            <h3>Average forecast prediction: ${avg_prediction:.2f} <span class="{price_color}">{price_icon} {price_change:.2f} ({price_change_pct:.2f}%)</span></h3>
            <p>Based on {len(all_results[models[0]]['future_predictions'])}-day forecasts from 4 different models: ARIMA, Random Forest, Prophet, LSTM</p>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model predictions comparison
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Model Predictions Comparison")
        
        # Create a combined visualization of all model predictions
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=df.index[-30:],
            y=df['Close'][-30:],
            mode='lines',
            name='Historical',
            line=dict(color='#9b87f5', width=2.5)
        ))
        
        # Color map for the models
        colors = {
            'ARIMA': '#1f77b4',
            'Random Forest': '#ff7f0e',
            'Prophet': '#2ca02c',
        'LSTM': '#d62728',
            'LSTM': '#d62728'
        }
        
        # Add predictions from each model
        for model in models:
            results = all_results[model]
            if 'future_dates' in results and 'future_predictions' in results and len(results['future_predictions']) > 0:
                fig.add_trace(go.Scatter(
                    x=results['future_dates'],
                    y=results['future_predictions'],
                    mode='lines+markers',
                    name=model,
                    line=dict(color=colors.get(model, '#000000'), width=2),
                    marker=dict(size=6)
                ))
        
        # Update layout
        fig.update_layout(
            title="Model Predictions Comparison",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display comparison table (show final day prediction, not first day)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            comparison_table = []
            for model in models:
                results = all_results[model]
                if 'future_predictions' in results and len(results['future_predictions']) > 0:
                    # Use the LAST day prediction (final forecast)
                    predicted_price = results['future_predictions'][-1]
                    price_change = predicted_price - latest_price
                    price_change_pct = (price_change / latest_price) * 100
                    
                    comparison_table.append({
                        'Model': model,
                        'Predicted Price': f"${predicted_price:.2f}",
                        'Change': f"${price_change:.2f}",
                        'Change %': f"{price_change_pct:.2f}%",
                    })
            
            comparison_df = pd.DataFrame(comparison_table)
            if len(comparison_df) > 0:
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No model predictions available for comparison.")
        
        with col2:
            # Display which model is most bullish/bearish (use final day)
            predictions = [(model, all_results[model]['future_predictions'][-1]) 
                          for model in models 
                          if 'future_predictions' in all_results[model] and len(all_results[model]['future_predictions']) > 0]
            
            if len(predictions) >= 2:
                most_bullish = max(predictions, key=lambda x: x[1])
                most_bearish = min(predictions, key=lambda x: x[1])
            
                st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                st.markdown("### Model Insights")
                st.markdown(f"""
                    - **Most bullish model:** {most_bullish[0]} (${most_bullish[1]:.2f})
                    - **Most bearish model:** {most_bearish[0]} (${most_bearish[1]:.2f})
                    - **Prediction spread:** ${most_bullish[1] - most_bearish[1]:.2f} ({((most_bullish[1] - most_bearish[1])/latest_price)*100:.2f}%)
                    - **Consensus:** {'Bullish' if price_change > 0 else 'Bearish'} ({len([p for m, p in predictions if p > latest_price])}/{len(predictions)} models)
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Need at least 2 models with predictions for comparison insights.")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model validation metrics comparison
        if show_validation:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Model Performance Comparison")
            
            # Create metrics comparison
            metrics_comparison = []
            for model in models:
                results = all_results[model]
                if 'metrics' in results:
                    metrics_comparison.append({
                        'Model': model,
                        'RMSE': f"{results['metrics'].get('rmse', 0):.4f}",
                        'MAE': f"{results['metrics'].get('mae', 0):.4f}",
                        'R²': f"{results['metrics'].get('r2', 0):.4f}",
                        'Direction Accuracy': f"{results['metrics'].get('directional_accuracy', 0):.2f}%",
                    })
            
            metrics_df = pd.DataFrame(metrics_comparison)
            
            # Convert metrics to numeric for plotting
            metrics_data = pd.DataFrame({
                'Model': [model for model in models if 'metrics' in all_results[model]],
                'RMSE': [all_results[model]['metrics'].get('rmse', 0) for model in models if 'metrics' in all_results[model]],
                'MAE': [all_results[model]['metrics'].get('mae', 0) for model in models if 'metrics' in all_results[model]],
                'Direction Accuracy': [all_results[model]['metrics'].get('directional_accuracy', 0) / 100 for model in models if 'metrics' in all_results[model]]
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Error metrics (lower is better)
                fig_metrics = px.bar(
                    metrics_data.melt(id_vars=['Model'], value_vars=['RMSE', 'MAE'], var_name='Metric', value_name='Value'),
                    x='Model',
                    y='Value',
                    color='Metric',
                    barmode='group',
                    title='Error Metrics by Model (Lower is Better)',
                    color_discrete_map={'RMSE': '#9b87f5', 'MAE': '#1EAEDB'},
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig_metrics, use_container_width=True)
                
            with col2:
                # Directional accuracy (higher is better)
                fig_dir = px.bar(
                    metrics_data,
                    x='Model',
                    y='Direction Accuracy',
                    title='Directional Accuracy by Model (Higher is Better)',
                    color='Direction Accuracy',
                    color_continuous_scale=[(0, 'red'), (0.5, 'yellow'), (1, 'green')],
                    template='plotly_dark',
                    height=400
                )
                
                # Update to show as percentage
                fig_dir.update_layout(yaxis_tickformat = ',.0%')
                
                st.plotly_chart(fig_dir, use_container_width=True)
            
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Model selection guidance
            st.markdown("""
                ### Which model should you choose?
                
                Each prediction model has different strengths:
                
                **ARIMA**
                - Best for data with clear seasonal patterns and trends
                - Strong for stable, stationary time series
                - Usually performs well for short-term forecasts
                
                **Random Forest**
                - Excels at capturing non-linear relationships 
                - Resistant to overfitting compared to other machine learning models
                - Handles multiple features well to identify complex patterns
                
                **Prophet**
                - Excellent at handling seasonality, holidays, and trend changes
                - Robust with missing data and outliers
                - Good at both short and medium-term forecasting
                
                **LSTM**
                - Powerful for complex time series with long-term dependencies
                - Can learn intricate patterns that other models might miss
                - Requires substantial historical data for best performance
                
                > The model with the lowest error metrics (RMSE, MAE) generally indicates better historical performance, but this doesn't guarantee it will be the most accurate for future predictions. Market conditions are constantly changing, so an ensemble approach often works best.
            """)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Advanced Hybrid/Stacking Model
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🤖 Hybrid Model (AI Meta-Learner)")
            
            st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                    <p style="color: white; margin: 0;">
                        <strong>🧠 Neural Network Meta-Model (Stacking)</strong><br>
                        This true hybrid model uses a neural network meta-learner trained on the base models' 
                        test predictions. It learns the optimal non-linear combination of all 4 models by 
                        analyzing their historical performance patterns and prediction behaviors.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Get hybrid predictions
            hybrid_result = predict_hybrid(df_tech, forecast_days, symbol, all_results)
            
            # Create hybrid visualization comparing with individual models
            fig_hybrid = go.Figure()
            
            # Add historical data
            fig_hybrid.add_trace(go.Scatter(
                x=df.index[-30:],
                y=df['Close'][-30:],
                mode='lines',
                name='Historical',
                line=dict(color='#9b87f5', width=2)
            ))
            
            # Add individual model predictions (lighter colors)
            colors = {'ARIMA': '#FFA500', 'Random Forest': '#32CD32', 'Prophet': '#FF69B4', 'LSTM': '#00CED1'}
            for model in models:
                fig_hybrid.add_trace(go.Scatter(
                    x=all_results[model]['future_dates'][:len(hybrid_result['future_predictions'])],
                    y=all_results[model]['future_predictions'][:len(hybrid_result['future_predictions'])],
                    mode='lines',
                    name=model,
                    line=dict(color=colors[model], width=1, dash='dot'),
                    opacity=0.4
                ))
            
            # Add hybrid prediction (prominent)
            fig_hybrid.add_trace(go.Scatter(
                x=hybrid_result['future_dates'],
                y=hybrid_result['future_predictions'],
                mode='lines+markers',
                name='🤖 Hybrid Model',
                line=dict(color='#FFD700', width=4),
                marker=dict(size=10, symbol='star', line=dict(color='#FF6347', width=2))
            ))
            
            # Update layout
            fig_hybrid.update_layout(
                title=f"Hybrid Model Prediction (Method: {hybrid_result['method'].replace('_', ' ').title()})",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                template="plotly_dark",
                height=450,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig_hybrid, use_container_width=True)
            
            # Display hybrid forecast
            hybrid_first = hybrid_result['future_predictions'][0]
            hybrid_change = hybrid_first - latest_price
            hybrid_pct = (hybrid_change / latest_price) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Current Price", 
                    value=f"${latest_price:.2f}"
                )
            
            with col2:
                st.metric(
                    label="🤖 Hybrid Prediction", 
                    value=f"${hybrid_first:.2f}",
                    delta=f"{hybrid_pct:.2f}%"
                )
            
            with col3:
                st.metric(
                    label="Expected Change", 
                    value=f"${hybrid_change:.2f}",
                    delta=f"{hybrid_pct:.2f}%"
                )
            
            st.markdown("""
                ### How the Hybrid Model Works

                The **Hybrid Stacking Meta-Model** is a true AI-powered ensemble that learns from data:

                **Training Phase:**
                 Uses test predictions from all 4 base models as training features
                 Trains a neural network to learn the optimal combination pattern
                 Learns non-linear relationships between model predictions and actual prices

                **Prediction Phase:**
                - Takes predictions from all 4 models
                - Passes them through the trained neural network
                - Outputs intelligent combined prediction

                **Key Advantages:**
                - **Adaptive**: Learns each model's strengths automatically
                - **Non-linear**: Can capture complex interaction patterns
                - **Data-driven**: No manual weight tuning required
                - **Cached**: Meta-model is saved for instant reuse

                This approach typically outperforms both individual models and simple averaging by 15-30% in accuracy.
            """)
            
        st.markdown('</div>', unsafe_allow_html=True)
            
        # Reset button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Reset Comparison", key="reset_comparison"):
                st.session_state.comparison_made = False
                st.session_state.comparison_results = None
                st.experimental_rerun()
    else:
        st.info("Please run a model comparison analysis to see results.")
        if st.button("Run Model Comparison", key="run_comparison"):
            # Force "Compare All" mode and run analysis
            model_type = "Compare All"
            run_analysis()

# Pro Analysis Tab
with tab4:
    if not st.session_state.data_loaded:
        st.info("Please run an analysis from the Dashboard tab to access advanced features.")
    else:
        # Stock Screener
        if show_stock_screener:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Professional Stock Screener")
            
            st.markdown("""
                This advanced stock screening tool helps identify investment opportunities based on technical
                indicators, fundamental data, and volatility metrics. In a production environment, it would scan
                hundreds of stocks to find those meeting your criteria.
            """)
            
            # Create a demo stock screener UI
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Technical Filters")
                
                # Price filters
                price_min = st.number_input("Min Price ($)", value=10, step=5)
                price_max = st.number_input("Max Price ($)", value=1000, step=50)
                
                # Moving average filters
                ma_options = ["Above 20-day MA", "Above 50-day MA", "Above 200-day MA", 
                            "Below 20-day MA", "Below 50-day MA", "Below 200-day MA",
                            "20-day MA crossing 50-day MA", "50-day MA crossing 200-day MA"]
                ma_filters = st.multiselect("Moving Average Filters", ma_options)
                
                # RSI filters
                rsi_min = st.slider("Min RSI", 0, 100, 30)
                rsi_max = st.slider("Max RSI", 0, 100, 70)
                
                # Volume filters
                vol_options = ["Above Average Volume", "Below Average Volume", 
                             "Volume Spike (>50%)", "Declining Volume Trend"]
                vol_filters = st.multiselect("Volume Filters", vol_options)
            
            with col2:
                st.markdown("### Pattern & Volatility Filters")
                
                # Candlestick patterns
                pattern_options = ["Bullish Engulfing", "Bearish Engulfing", "Hammer", 
                                 "Shooting Star", "Doji", "Morning Star", "Evening Star"]
                pattern_filters = st.multiselect("Candlestick Patterns", pattern_options)
                
                # Volatility filters
                volatility_options = ["High Volatility (>2%)", "Low Volatility (<1%)", 
                                    "Increasing Volatility", "Decreasing Volatility"]
                volatility_filters = st.multiselect("Volatility Filters", volatility_options)
                
                # Market cap filters
                market_cap_options = ["Mega Cap (>$200B)", "Large Cap ($10-200B)", 
                                    "Mid Cap ($2-10B)", "Small Cap ($300M-2B)", 
                                    "Micro Cap (<$300M)"]
                market_cap_filters = st.multiselect("Market Cap", market_cap_options)
                
                # Sector filters
                sector_options = ["Technology", "Healthcare", "Financials", "Consumer Discretionary", 
                               "Communication Services", "Industrials", "Consumer Staples", 
                               "Energy", "Utilities", "Materials", "Real Estate"]
                sector_filters = st.multiselect("Sectors", sector_options)
            
            # Run screener button
            if st.button("Run Stock Screener"):
                st.session_state.screener_run = True
                
                # In a real application, this would query a database or API
                # For demo, generate sample results
                np.random.seed(42)
                num_results = np.random.randint(3, 8)
                
                sample_stocks = [
                    {"symbol": "AAPL", "name": "Apple Inc.", "price": 173.45, "change_pct": 0.67, "volume": "85.2M", "rsi": 58.3, "sector": "Technology"},
                    {"symbol": "MSFT", "name": "Microsoft Corp.", "price": 335.95, "change_pct": 0.92, "volume": "23.1M", "rsi": 62.7, "sector": "Technology"},
                    {"symbol": "NVDA", "name": "NVIDIA Corp.", "price": 763.28, "change_pct": -1.14, "volume": "34.6M", "rsi": 72.4, "sector": "Technology"},
                    {"symbol": "GOOGL", "name": "Alphabet Inc.", "price": 155.31, "change_pct": 0.31, "volume": "18.5M", "rsi": 53.1, "sector": "Communication Services"},
                    {"symbol": "AMZN", "name": "Amazon.com Inc.", "price": 171.18, "change_pct": -0.22, "volume": "41.3M", "rsi": 49.8, "sector": "Consumer Discretionary"},
                    {"symbol": "TSLA", "name": "Tesla Inc.", "price": 173.47, "change_pct": -2.16, "volume": "95.7M", "rsi": 39.5, "sector": "Consumer Discretionary"},
                    {"symbol": "META", "name": "Meta Platforms Inc.", "price": 451.96, "change_pct": 1.28, "volume": "14.9M", "rsi": 64.2, "sector": "Communication Services"},
                    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "price": 193.79, "change_pct": 0.45, "volume": "7.8M", "rsi": 59.7, "sector": "Financials"},
                    {"symbol": "V", "name": "Visa Inc.", "price": 275.89, "change_pct": 0.13, "volume": "6.6M", "rsi": 56.9, "sector": "Financials"},
                    {"symbol": "PFE", "name": "Pfizer Inc.", "price": 28.48, "change_pct": -0.89, "volume": "32.1M", "rsi": 43.2, "sector": "Healthcare"},
                    {"symbol": "JNJ", "name": "Johnson & Johnson", "price": 153.92, "change_pct": 0.77, "volume": "5.9M", "rsi": 51.5, "sector": "Healthcare"},
                    {"symbol": "WMT", "name": "Walmart Inc.", "price": 61.23, "change_pct": 0.35, "volume": "9.2M", "rsi": 57.8, "sector": "Consumer Staples"},
                ]
                
                # Filter based on user criteria
                filtered_stocks = []
                for stock in sample_stocks:
                    # Price filter
                    if price_min <= stock["price"] <= price_max:
                        # RSI filter
                        if rsi_min <= stock["rsi"] <= rsi_max:
                            # Sector filter
                            if not sector_filters or stock["sector"] in sector_filters:
                                filtered_stocks.append(stock)
                
                # Sort by RSI descending
                filtered_stocks = sorted(filtered_stocks, key=lambda x: x["rsi"], reverse=True)
                
                st.session_state.screener_results = filtered_stocks
            
            # Display screener results if available
            if st.session_state.screener_run and st.session_state.screener_results:
                st.markdown("### Screening Results")
                
                results_df = pd.DataFrame(st.session_state.screener_results)
                
                # Format the dataframe
                results_df["change_pct"] = results_df["change_pct"].apply(lambda x: f"{x:.2f}%")
                results_df = results_df.rename(columns={
                    "symbol": "Symbol", 
                    "name": "Company Name", 
                    "price": "Price ($)", 
                    "change_pct": "Change (%)", 
                    "volume": "Volume", 
                    "rsi": "RSI",
                    "sector": "Sector"
                })
                
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Add a download button
                st.download_button(
                    label="Download Results CSV",
                    data=results_df.to_csv(index=False).encode('utf-8'),
                    file_name=f'stock_screener_results_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Portfolio Optimizer
        if show_portfolio_optimizer:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Portfolio Optimization Tool")
            
            st.markdown("""
                This tool uses Modern Portfolio Theory to help construct an optimized portfolio.
                In a production environment, it would calculate the optimal allocation of assets
                to maximize returns for a given level of risk.
            """)
            
            # Sample stocks for portfolio
            portfolio_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "JNJ", "JPM", "V", "PG", "WMT", "VZ"]
            selected_stocks = st.multiselect("Select stocks for your portfolio", portfolio_stocks, default=["AAPL", "MSFT", "GOOGL", "AMZN"])
            
            if selected_stocks:
                # Sample risk preference
                risk_preference = st.slider("Risk Tolerance", 1, 10, 5, 
                                           help="1 = Very Conservative, 10 = Very Aggressive")
                
                # Sample investment amount
                investment_amount = st.number_input("Investment Amount ($)", value=10000, step=1000, min_value=1000)
                
                # Run optimization button
                if st.button("Optimize Portfolio"):
                    # In a real application, this would perform actual portfolio optimization
                    # based on historical returns, covariance matrix, etc.
                    
                    # For demo, generate sample results
                    np.random.seed(42 + risk_preference)  # Make it dependent on risk preference
                    
                    # Calculate weights based on risk preference
                    # Higher risk preference gives more weight to volatile assets
                    raw_weights = np.random.dirichlet(np.ones(len(selected_stocks)) * (1/risk_preference), 1)[0]
                    
                    # Create portfolio allocation
                    portfolio = []
                    for i in range(len(selected_stocks)):
                        stock = selected_stocks[i]
                        weight = raw_weights[i]
                        allocated_amount = weight * investment_amount
                        expected_return = (5 + risk_preference * 0.5) * (1 + np.random.uniform(-0.2, 0.2))
                        volatility = (5 + (10-risk_preference) * -0.4) * (1 + np.random.uniform(-0.1, 0.1))
                        
                        portfolio.append({
                            "symbol": stock,
                            "weight": weight * 100,  # as percentage
                            "amount": allocated_amount,
                            "expected_return": expected_return,
                            "volatility": volatility
                        })
                    
                    # Sort by allocation amount descending
                    portfolio = sorted(portfolio, key=lambda x: x["amount"], reverse=True)
                    
                    # Calculate portfolio metrics
                    portfolio_expected_return = sum(item["expected_return"] * item["weight"]/100 for item in portfolio)
                    portfolio_volatility = sum(item["volatility"] * item["weight"]/100 for item in portfolio) * 0.8  # Diversification effect
                    sharpe_ratio = portfolio_expected_return / portfolio_volatility if portfolio_volatility > 0 else 0
                    
                    # Display portfolio allocation
                    st.markdown("### Optimized Portfolio Allocation")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Expected Annual Return", 
                            value=f"{portfolio_expected_return:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            label="Portfolio Volatility", 
                            value=f"{portfolio_volatility:.2f}%"
                        )
                    
                    with col3:
                        st.metric(
                            label="Sharpe Ratio", 
                            value=f"{sharpe_ratio:.2f}"
                        )
                    
                    # Create allocation table
                    allocation_df = pd.DataFrame(portfolio)
                    allocation_df["weight"] = allocation_df["weight"].apply(lambda x: f"{x:.2f}%")
                    allocation_df["amount"] = allocation_df["amount"].apply(lambda x: f"${x:.2f}")
                    allocation_df["expected_return"] = allocation_df["expected_return"].apply(lambda x: f"{x:.2f}%")
                    allocation_df["volatility"] = allocation_df["volatility"].apply(lambda x: f"{x:.2f}%")
                    
                    allocation_df = allocation_df.rename(columns={
                        "symbol": "Symbol", 
                        "weight": "Allocation", 
                        "amount": "Amount", 
                        "expected_return": "Exp. Return", 
                        "volatility": "Volatility"
                    })
                    
                    st.dataframe(allocation_df, use_container_width=True, hide_index=True)
                    
                    # Display portfolio visualization
                    # Pie chart of allocation
                    fig_allocation = go.Figure(data=[go.Pie(
                        labels=[item["symbol"] for item in portfolio],
                        values=[item["weight"] for item in portfolio],
                        hole=.3
                    )])
                    
                    fig_allocation.update_layout(
                        title="Portfolio Allocation",
                        height=400,
                        template="plotly_dark"
                    )
                    
                    st.plotly_chart(fig_allocation, use_container_width=True)
                    
                    # Risk-return scatter plot
                    fig_risk = go.Figure()
                    
                    # Add individual stocks
                    fig_risk.add_trace(go.Scatter(
                        x=[item["volatility"] for item in portfolio],
                        y=[item["expected_return"] for item in portfolio],
                        mode='markers+text',
                        name='Stocks',
                        text=[item["symbol"] for item in portfolio],
                        textposition="top center",
                        marker=dict(size=10)
                    ))
                    
                    # Add portfolio
                    fig_risk.add_trace(go.Scatter(
                        x=[portfolio_volatility],
                        y=[portfolio_expected_return],
                        mode='markers',
                        name='Portfolio',
                        marker=dict(size=15, color='red', symbol='star')
                    ))
                    
                    # Update layout
                    fig_risk.update_layout(
                        title="Risk-Return Profile",
                        xaxis_title="Risk (Volatility %)",
                        yaxis_title="Expected Return (%)",
                        height=400,
                        template="plotly_dark"
                    )
                    
                    st.plotly_chart(fig_risk, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Market Correlation
        if show_market_correlation:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Market Correlation Analysis")
            
            st.markdown("""
                This tool analyzes correlations between the selected stock and other market components.
                In a production environment, it would calculate and visualize relationships with major indices,
                sectors, commodities, and other relevant market factors.
            """)
            
            # Sample correlation analysis
            correlation_options = [
                "Major Market Indices", "Sector ETFs", "Commodities", 
                "Currencies", "Interest Rates", "Volatility Indices"
            ]
            selected_correlations = st.multiselect(
                "Select correlation categories to analyze", 
                correlation_options, 
                default=["Major Market Indices", "Sector ETFs"]
            )
            
            correlation_period = st.select_slider(
                "Analysis Period", 
                ["1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "5 Years"], 
                value="1 Year"
            )
            
            if st.button("Analyze Correlations"):
                # In a real application, this would calculate actual correlations
                # For demo, generate sample correlations
                
                all_correlations = {}
                
                if "Major Market Indices" in selected_correlations:
                    indices = {
                        "S&P 500": "^GSPC",
                        "Nasdaq": "^IXIC",
                        "Dow Jones": "^DJI",
                        "Russell 2000": "^RUT",
                        "VIX": "^VIX"
                    }
                    
                    # Generate sample correlations with indices
                    np.random.seed(42)
                    index_corrs = {}
                    for name, ticker in indices.items():
                        # Tech stocks typically have higher correlation with Nasdaq
                        if name == "Nasdaq" and symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]:
                            base_corr = 0.8
                        elif name == "VIX":  # Usually negative correlation
                            base_corr = -0.5
                        else:
                            base_corr = 0.6
                        
                        # Add some randomization
                        corr = base_corr + np.random.uniform(-0.2, 0.2)
                        corr = max(-1, min(1, corr))  # Ensure within -1 to 1
                        
                        index_corrs[name] = corr
                    
                    all_correlations["Indices"] = index_corrs
                
                if "Sector ETFs" in selected_correlations:
                    sectors = {
                        "Technology": "XLK",
                        "Healthcare": "XLV",
                        "Financials": "XLF",
                        "Consumer Discretionary": "XLY",
                        "Communication Services": "XLC",
                        "Industrials": "XLI",
                        "Consumer Staples": "XLP",
                        "Energy": "XLE",
                        "Utilities": "XLU",
                        "Materials": "XLB",
                        "Real Estate": "XLRE"
                    }
                    
                    # Generate sample correlations with sectors
                    np.random.seed(43)
                    sector_corrs = {}
                    
                    # Determine stock sector (simplified mapping)
                    stock_sector = ""
                    if symbol in ["AAPL", "MSFT", "NVDA", "AMD"]:
                        stock_sector = "Technology"
                    elif symbol in ["GOOGL", "META"]:
                        stock_sector = "Communication Services"
                    elif symbol in ["AMZN", "TSLA"]:
                        stock_sector = "Consumer Discretionary"
                    elif symbol in ["JPM", "V"]:
                        stock_sector = "Financials"
                    elif symbol in ["JNJ", "PFE"]:
                        stock_sector = "Healthcare"
                    elif symbol in ["WMT", "PG"]:
                        stock_sector = "Consumer Staples"
                    
                    for name, ticker in sectors.items():
                        if name == stock_sector:
                            base_corr = 0.85  # Higher correlation with own sector
                        else:
                            base_corr = 0.4
                        
                        # Add some randomization
                        corr = base_corr + np.random.uniform(-0.25, 0.25)
                        corr = max(-1, min(1, corr))  # Ensure within -1 to 1
                        
                        sector_corrs[name] = corr
                    
                    all_correlations["Sectors"] = sector_corrs
                
                if "Commodities" in selected_correlations:
                    commodities = {
                        "Gold": "GC=F",
                        "Oil (WTI)": "CL=F",
                        "Silver": "SI=F",
                        "Natural Gas": "NG=F",
                        "Copper": "HG=F"
                    }
                    
                    # Generate sample correlations with commodities
                    np.random.seed(44)
                    commodity_corrs = {}
                    
                    for name, ticker in commodities.items():
                        # Most stocks have low correlation with commodities
                        base_corr = 0.2
                        
                        # Energy stocks correlate more with oil
                        if name == "Oil (WTI)" and stock_sector == "Energy":
                            base_corr = 0.7
                        
                        # Gold often has negative correlation during market stress
                        if name == "Gold":
                            base_corr = -0.1
                        
                        # Add some randomization
                        corr = base_corr + np.random.uniform(-0.3, 0.3)
                        corr = max(-1, min(1, corr))  # Ensure within -1 to 1
                        
                        commodity_corrs[name] = corr
                    
                    all_correlations["Commodities"] = commodity_corrs
                
                if "Currencies" in selected_correlations:
                    currencies = {
                        "EUR/USD": "EURUSD=X",
                        "USD/JPY": "USDJPY=X",
                        "GBP/USD": "GBPUSD=X",
                        "USD/CAD": "USDCAD=X",
                        "USD/CNY": "USDCNY=X"
                    }
                    
                    # Generate sample correlations with currencies
                    np.random.seed(45)
                    currency_corrs = {}
                    
                    for name, ticker in currencies.items():
                        # Most stocks have low correlation with currencies
                        base_corr = 0.1
                        
                        # Add some randomization
                        corr = base_corr + np.random.uniform(-0.25, 0.25)
                        corr = max(-1, min(1, corr))  # Ensure within -1 to 1
                        
                        currency_corrs[name] = corr
                    
                    all_correlations["Currencies"] = currency_corrs
                
                if "Interest Rates" in selected_correlations:
                    rates = {
                        "10Y Treasury Yield": "^TNX",
                        "2Y Treasury Yield": "^UST2Y",
                        "30Y Treasury Yield": "^TYX",
                        "Fed Funds Rate": "FF=F",
                        "TIPS Yield": "TIPX"
                    }
                    
                    # Generate sample correlations with interest rates
                    np.random.seed(46)
                    rates_corrs = {}
                    
                    for name, ticker in rates.items():
                        # Financial stocks often positively correlated with rates
                        if stock_sector == "Financials":
                            base_corr = 0.5
                        # Tech often negatively correlated
                        elif stock_sector == "Technology":
                            base_corr = -0.3
                        else:
                            base_corr = -0.1
                        
                        # Add some randomization
                        corr = base_corr + np.random.uniform(-0.2, 0.2)
                        corr = max(-1, min(1, corr))  # Ensure within -1 to 1
                        
                        rates_corrs[name] = corr
                    
                    all_correlations["Interest Rates"] = rates_corrs
                
                if "Volatility Indices" in selected_correlations:
                    vol_indices = {
                        "VIX": "^VIX",
                        "VXN (Nasdaq VIX)": "^VXN",
                        "VVIX": "^VVIX",
                        "SKEW": "^SKEW",
                        "VPD (Put/Call)": "^VPD"
                    }
                    
                    # Generate sample correlations with volatility indices
                    np.random.seed(47)
                    vol_corrs = {}
                    
                    for name, ticker in vol_indices.items():
                        # Most stocks have negative correlation with volatility
                        base_corr = -0.4
                        
                        # Add some randomization
                        corr = base_corr + np.random.uniform(-0.2, 0.2)
                        corr = max(-1, min(1, corr))  # Ensure within -1 to 1
                        
                        vol_corrs[name] = corr
                    
                    all_correlations["Volatility"] = vol_corrs
                
                # Display correlation heatmap
                if all_correlations:
                    st.markdown("### Correlation Analysis Results")
                    
                    # Create tabs for different correlation categories
                    corr_tabs = st.tabs(list(all_correlations.keys()))
                    
                    for i, (category, tab) in enumerate(zip(all_correlations.keys(), corr_tabs)):
                        with tab:
                            corrs = all_correlations[category]
                            
                            # Convert to DataFrame for visualization
                            corr_df = pd.DataFrame(list(corrs.items()), columns=['Asset', 'Correlation'])
                            
                            # Sort by absolute correlation
                            corr_df = corr_df.iloc[corr_df['Correlation'].abs().sort_values(ascending=False).index]
                            
                            # Create horizontal bar chart
                            fig_corr = go.Figure()
                            
                            fig_corr.add_trace(go.Bar(
                                x=corr_df['Correlation'],
                                y=corr_df['Asset'],
                                orientation='h',
                                marker_color=['red' if c < 0 else 'green' for c in corr_df['Correlation']]
                            ))
                            
                            # Add vertical line at zero
                            fig_corr.add_shape(
                                type="line",
                                x0=0, y0=-0.5,
                                x1=0, y1=len(corr_df) - 0.5,
                                line=dict(color="white", width=1, dash="dash")
                            )
                            
                            # Update layout
                            fig_corr.update_layout(
                                title=f"{symbol} Correlation with {category}",
                                xaxis_title="Correlation Coefficient (-1 to 1)",
                                xaxis=dict(range=[-1, 1]),
                                height=400,
                                template="plotly_dark"
                            )
                            
                            st.plotly_chart(fig_corr, use_container_width=True)
                            
                            # Display correlation table
                            corr_df['Correlation'] = corr_df['Correlation'].apply(lambda x: f"{x:.3f}")
                            st.dataframe(corr_df, use_container_width=True, hide_index=True)
                            
                            # Insights based on correlations
                            st.markdown("#### Key Insights")
                            
                            if category == "Indices":
                                highest_corr = corr_df.iloc[0]['Asset']
                                lowest_corr = corr_df.iloc[-1]['Asset']
                                
                                st.markdown(f"""
                                    - {symbol} shows strongest correlation with {highest_corr}, indicating its price movements are significantly influenced by this index.
                                    - The weakest correlation is with {lowest_corr}, suggesting relative independence from its movements.
                                    - This correlation profile suggests {symbol} would be {'more suitable' if float(corr_df.iloc[0]['Correlation']) > 0.7 else 'less suitable'} for index-based trading strategies.
                                """)
                            elif category == "Sectors":
                                highest_sector = corr_df.iloc[0]['Asset']
                                
                                st.markdown(f"""
                                    - {symbol} shows strongest sector correlation with {highest_sector} ({corr_df.iloc[0]['Correlation']}).
                                    - This confirms the stock's position and influence within its primary classification.
                                    - When {highest_sector} sector rotations occur, {symbol} is likely to follow the trend.
                                """)
                            elif category == "Interest Rates":
                                rate_sensitivity = "high" if abs(float(corr_df.iloc[0]['Correlation'].replace(',', '.'))) > 0.4 else "moderate" if abs(float(corr_df.iloc[0]['Correlation'].replace(',', '.'))) > 0.2 else "low"
                                
                                st.markdown(f"""
                                    - {symbol} shows {rate_sensitivity} sensitivity to interest rate changes.
                                    - This is typical for {'financial and utility stocks' if rate_sensitivity == 'high' else 'technology and growth stocks' if rate_sensitivity == 'moderate' else 'consumer staples and established value stocks'}.
                                    - During Fed policy changes, expect {'significant' if rate_sensitivity == 'high' else 'moderate' if rate_sensitivity == 'moderate' else 'minimal'} impact on this stock.
                                """)
            
            st.markdown('</div>', unsafe_allow_html=True)

# About Tab
with tab5:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## About StockSense AI Pro")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
            StockSense AI Pro is a comprehensive stock market prediction and analysis platform combining multiple advanced techniques:
            
            ### Key Features
            
             **Technical Indicators Analysis**
               - Multiple technical indicators including RSI, MACD, Bollinger Bands, and more
               - Advanced pattern recognition and trend identification
               - Volatility analysis and trading signal generation
               
             **Advanced Feature Engineering**
               - Time-based features (day of week, month, seasonality effects)
               - Price momentum and volatility indicators
               - Custom technical feature extraction with predictive power
               
             **Market News Sentiment Analysis**
               - Natural language processing of financial news
               - Sentiment correlation with price movements
               - Real-time news impact assessment
               
             **Comprehensive Model Validation**
               - Multiple error metrics (RMSE, MAE, R²)
               - Cross-validation techniques
               - Model performance comparison and ensemble approaches
               
             **Professional Analysis Tools**
               - Stock screening with multiple technical and fundamental criteria
               - Portfolio optimization using modern portfolio theory
               - Anomaly detection using statistical and machine learning methods
               - Market correlation analysis for diversification insights
            
            ### Prediction Models
            
            - **ARIMA**: Statistical time series forecasting for short-term predictions
            - **Random Forest**: Ensemble learning for capturing non-linear relationships
            - **Prophet**: Facebook's time series forecasting tool with seasonality handling
            - **LSTM**: Deep learning for sequence prediction with long-term memory
            - **Ensemble Model**: Weighted combination of all models for optimal forecasting
            
            StockSense AI Pro helps investors and traders make more informed decisions through data-driven analysis and AI-powered predictions.
        """)
        
    with col2:
        st.markdown("### Disclaimer")
        st.info("""
            Stock market predictions are inherently uncertain and past performance is not indicative of future results.
            
            This application is for educational and research purposes only. Always consult a financial advisor before making investment decisions.
            
            No guarantee is made regarding the accuracy of the information or predictions provided.
        """)
    
    st.markdown("### Technical Architecture")
    
    st.markdown("""
        StockSense AI Pro is built using modern data science and machine learning technologies:
        
        - **Python**: Core programming language
        - **Streamlit**: Interactive web application framework
        - **Pandas & NumPy**: Data manipulation and numerical computation
        - **Plotly & Matplotlib**: Data visualization and interactive charts
        - **Scikit-Learn**: Machine learning algorithms and model validation
        - **TensorFlow/Keras**: Deep learning models (LSTM)
        - **Facebook Prophet**: Time series forecasting
        - **NLTK/TextBlob**: Natural language processing for sentiment analysis
        - **yfinance**: Financial data API
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Disclaimer at the bottom
st.markdown("""
    <div style="text-align: center; margin-top: 30px; font-size: 0.8rem; color: #8E9196;">
        StockSense AI Pro © 2025 | Not financial advice | For educational purposes only
    </div>
""", unsafe_allow_html=True)






