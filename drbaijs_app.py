# ============================================================
# INTRADAY STOCK PREDICTOR - ENHANCED WITH VWAP & MORE
# All Indian Stocks - Python 3.14 Compatible - FIXED
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
import warnings
warnings.filterwarnings('ignore')
# Install: pip install nsepy
import nsepy

@st.cache_data(ttl=86400)
def get_nse_stock_list():
    """Get ALL NSE stocks using nsepy"""
    try:
        from nsepy import get_history
        from nsepy.helpers import get_equity_list
        stocks = get_equity_list()
        return sorted(stocks)
    except:
        return get_comprehensive_stock_list()  # Fallback list
# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Intraday Stock Predictor - India",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# CUSTOM CSS
# ---------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #00ff88;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .buy-signal {
        background: #00ff88;
        color: #000;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 2.5rem;
    }
    .wait-signal {
        background: #ffaa00;
        color: #000;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 2.5rem;
    }
    .avoid-signal {
        background: #ff4444;
        color: #fff;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 2.5rem;
    }
    .stocks-count {
        background: #1a1a2e;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 1px solid #333;
        text-align: center;
        font-size: 0.9rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 1. STOCK LIST FUNCTIONS
# ---------------------------
@st.cache_data(ttl=86400)
def get_comprehensive_stock_list():
    """Extended stock list with all major stocks"""
    return sorted(set([
        # NIFTY 50
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'ITC', 'HINDUNILVR', 'SBIN', 'BHARTIARTL', 'KOTAKBANK',
        'LT', 'WIPRO', 'HCLTECH', 'ASIANPAINT', 'AXISBANK',
        'MARUTI', 'BAJFINANCE', 'TATAMOTORS', 'M&M', 'TATASTEEL',
        'NTPC', 'ONGC', 'POWERGRID', 'ADANIPORTS', 'COALINDIA',
        'NESTLEIND', 'ULTRACEMCO', 'GRASIM', 'JSWSTEEL', 'TITAN',
        'BRITANNIA', 'HDFCLIFE', 'SBILIFE', 'INDUSINDBK', 'BAJAJFINSV',
        'HINDALCO', 'APOLLOHOSP', 'DRREDDY', 'CIPLA', 'DIVISLAB',
        'SUNPHARMA', 'UPL', 'SHREECEM', 'TECHM', 'ADANIENT',
        
        # Your stocks
        'POLYCAB', 'BOSCHLTD', 'LLOYDSENGG', 'DIXON', 'HAL', 'ORKLAINDIA',
        
        # Add ALL missing stocks you want here
        'HINDUNILVR', 'MARICO', 'DABUR', 'BRITANNIA', 'NESTLEIND',
        'PIDILITIND', 'BERGEPAINT', 'ASIANPAINT', 'INDIGO', 'TATACONSUM',
        
        # BSE 100 and other major stocks
        'HDFCLIFE', 'SBILIFE', 'ICICIPRULI', 'HDFCAMC', 'MUTHOOTFIN',
        'CHOLAFIN', 'BAJAJFINSV', 'BAJFINANCE', 'LICHSGFIN', 'PFC', 'RECLTD',
        'PEL', 'SRF', 'ATUL', 'DEEPAKNTR', 'PIDILITIND',
        'ABB', 'SIEMENS', 'VOLTAS', 'CROMPTON', 'HAVELLS',
        'BEL', 'BHEL', 'NMDC', 'GAIL', 'IOC', 'BPCL', 'HINDPETRO',
        'HEROMOTOCO', 'BAJAJ-AUTO', 'EICHERMOT', 'TIINDIA', 'ESCORTS',
        'DLF', 'GODREJPROP', 'OBEROIRLTY', 'PHOENIXLTD',
        
        # Mid cap and small cap
        'IDEA', 'AIRTEL', 'JIOFIN', 'PAYTM', 'ZOMATO', 'SWIGGY',
        'MRF', 'APOLLOTYRE', 'CEATLTD', 'BALKRISIND', 'MOTHERSUMI',
        'LUPIN', 'BIOCON', 'TORNTPHARM', 'AUROPHARMA', 'GLENMARK',
        'CONCOR', 'ADANIGREEN', 'ADANITRANS', 'VEDL', 'JINDALSTEL',
        'SAIL', 'NATIONALUM', 'HINDZINC'
    ]
    return sorted(set(stocks))

def search_stocks(query, stock_list):
    if not query:
        return stock_list[:50]
    query_lower = query.lower()
    matches = [s for s in stock_list if query_lower in s.lower()]
    return matches[:100]

# ---------------------------
# 2. TECHNICAL INDICATORS WITH VWAP
# ---------------------------
def calculate_indicators(df):
    """Calculate comprehensive intraday indicators with VWAP"""
    data = df.copy()
    
    close = data['Close'].squeeze()
    high = data['High'].squeeze()
    low = data['Low'].squeeze()
    volume = data['Volume'].squeeze()
    
    # VWAP (Volume Weighted Average Price)
    data['vwap'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    data['vwap_dev'] = ((data['Close'] - data['vwap']) / data['vwap']) * 100
    
    # Returns
    data['ret_1'] = close.pct_change(1)
    data['ret_3'] = close.pct_change(3)
    data['ret_5'] = close.pct_change(5)
    data['ret_10'] = close.pct_change(10)
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    
    # Moving Averages
    data['ma_5'] = close.rolling(5).mean()
    data['ma_10'] = close.rolling(10).mean()
    data['ma_20'] = close.rolling(20).mean()
    data['ma_50'] = close.rolling(50).mean()
    
    # Price vs MA
    data['vs_ma5'] = ((close - data['ma_5']) / data['ma_5']) * 100
    data['vs_ma10'] = ((close - data['ma_10']) / data['ma_10']) * 100
    data['vs_ma20'] = ((close - data['ma_20']) / data['ma_20']) * 100
    
    # Bollinger Bands
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    data['bb_upper'] = ma20 + (2 * std20)
    data['bb_lower'] = ma20 - (2 * std20)
    data['bb_pct_b'] = (close - ma20) / (2 * std20)
    
    # Volume
    data['vol_ma_5'] = volume.rolling(5).mean()
    data['vol_ratio'] = volume / data['vol_ma_5']
    
    # Volatility
    data['volatility'] = data['ret_1'].rolling(10).std()
    data['volatility_annual'] = data['volatility'] * np.sqrt(252)
    
    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['atr'] = tr.rolling(14).mean()
    data['atr_pct'] = (data['atr'] / close) * 100
    
    # Stochastic
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    data['stoch_k'] = ((close - low_14) / (high_14 - low_14)) * 100
    data['stoch_d'] = data['stoch_k'].rolling(3).mean()
    
    # Price position
    data['price_position'] = ((close - low.rolling(20).min()) / 
                              (high.rolling(20).max() - low.rolling(20).min())) * 100
    
    # Support & Resistance
    data['resistance'] = high.rolling(20).max()
    data['support'] = low.rolling(20).min()
    
    # Target
    data['target'] = (close.shift(-1) > close).astype(int)
    
    return data

# ---------------------------
# 3. INDICATOR CATEGORIZATION
# ---------------------------
def categorize_indicators(data):
    """Categorize each indicator as Bullish, Bearish, or Neutral"""
    latest = data.iloc[-1]
    
    categories = {'Bullish': [], 'Bearish': [], 'Neutral': []}
    signals = []
    
    # Helper to safely get values
    def safe_get(key, default=50):
        try:
            val = float(latest[key])
            if pd.isna(val) or np.isinf(val):
                return default
            return val
        except:
            return default
    
    rsi = safe_get('rsi')
    macd_hist = safe_get('macd_hist')
    vs_ma5 = safe_get('vs_ma5')
    vs_ma20 = safe_get('vs_ma20')
    bb_pct_b = safe_get('bb_pct_b')
    vol_ratio = safe_get('vol_ratio')
    stoch_k = safe_get('stoch_k')
    price_pos = safe_get('price_position')
    vwap_dev = safe_get('vwap_dev')
    
    # RSI
    if rsi > 70:
        categories['Bearish'].append(f"RSI: {rsi:.1f} (Overbought)")
        signals.append({'indicator': 'RSI', 'value': rsi, 'signal': 'Bearish', 'weight': 2})
    elif rsi < 30:
        categories['Bullish'].append(f"RSI: {rsi:.1f} (Oversold)")
        signals.append({'indicator': 'RSI', 'value': rsi, 'signal': 'Bullish', 'weight': 2})
    else:
        categories['Neutral'].append(f"RSI: {rsi:.1f} (Neutral)")
        signals.append({'indicator': 'RSI', 'value': rsi, 'signal': 'Neutral', 'weight': 1})
    
    # MACD
    if macd_hist > 0:
        categories['Bullish'].append(f"MACD: {macd_hist:.2f} (Bullish)")
        signals.append({'indicator': 'MACD', 'value': macd_hist, 'signal': 'Bullish', 'weight': 2})
    else:
        categories['Bearish'].append(f"MACD: {macd_hist:.2f} (Bearish)")
        signals.append({'indicator': 'MACD', 'value': macd_hist, 'signal': 'Bearish', 'weight': 2})
    
    # VWAP Deviation
    if vwap_dev > 2:
        categories['Bearish'].append(f"VWAP: {vwap_dev:.1f}% (Above)")
        signals.append({'indicator': 'VWAP', 'value': vwap_dev, 'signal': 'Bearish', 'weight': 1.5})
    elif vwap_dev < -2:
        categories['Bullish'].append(f"VWAP: {vwap_dev:.1f}% (Below)")
        signals.append({'indicator': 'VWAP', 'value': vwap_dev, 'signal': 'Bullish', 'weight': 1.5})
    else:
        categories['Neutral'].append(f"VWAP: {vwap_dev:.1f}% (Near VWAP)")
        signals.append({'indicator': 'VWAP', 'value': vwap_dev, 'signal': 'Neutral', 'weight': 0.5})
    
    # MA5
    if vs_ma5 > 2:
        categories['Bullish'].append(f"Price vs MA5: {vs_ma5:.1f}% (Above)")
        signals.append({'indicator': 'MA5', 'value': vs_ma5, 'signal': 'Bullish', 'weight': 1})
    elif vs_ma5 < -2:
        categories['Bearish'].append(f"Price vs MA5: {vs_ma5:.1f}% (Below)")
        signals.append({'indicator': 'MA5', 'value': vs_ma5, 'signal': 'Bearish', 'weight': 1})
    else:
        categories['Neutral'].append(f"Price vs MA5: {vs_ma5:.1f}% (Neutral)")
        signals.append({'indicator': 'MA5', 'value': vs_ma5, 'signal': 'Neutral', 'weight': 0.5})
    
    # Bollinger
    if bb_pct_b > 0.8:
        categories['Bearish'].append(f"Bollinger: Upper Band")
        signals.append({'indicator': 'Bollinger', 'value': bb_pct_b, 'signal': 'Bearish', 'weight': 1.5})
    elif bb_pct_b < 0.2:
        categories['Bullish'].append(f"Bollinger: Lower Band")
        signals.append({'indicator': 'Bollinger', 'value': bb_pct_b, 'signal': 'Bullish', 'weight': 1.5})
    else:
        categories['Neutral'].append(f"Bollinger: Middle Band")
        signals.append({'indicator': 'Bollinger', 'value': bb_pct_b, 'signal': 'Neutral', 'weight': 0.5})
    
    # Volume
    if vol_ratio > 1.5:
        categories['Bullish'].append(f"Volume: {vol_ratio:.2f}x (High)")
        signals.append({'indicator': 'Volume', 'value': vol_ratio, 'signal': 'Bullish', 'weight': 1})
    elif vol_ratio < 0.5:
        categories['Bearish'].append(f"Volume: {vol_ratio:.2f}x (Low)")
        signals.append({'indicator': 'Volume', 'value': vol_ratio, 'signal': 'Bearish', 'weight': 1})
    else:
        categories['Neutral'].append(f"Volume: {vol_ratio:.2f}x (Normal)")
        signals.append({'indicator': 'Volume', 'value': vol_ratio, 'signal': 'Neutral', 'weight': 0.5})
    
    # Stochastic
    if stoch_k > 80:
        categories['Bearish'].append(f"Stochastic: {stoch_k:.1f} (Overbought)")
        signals.append({'indicator': 'Stochastic', 'value': stoch_k, 'signal': 'Bearish', 'weight': 1.5})
    elif stoch_k < 20:
        categories['Bullish'].append(f"Stochastic: {stoch_k:.1f} (Oversold)")
        signals.append({'indicator': 'Stochastic', 'value': stoch_k, 'signal': 'Bullish', 'weight': 1.5})
    else:
        categories['Neutral'].append(f"Stochastic: {stoch_k:.1f} (Neutral)")
        signals.append({'indicator': 'Stochastic', 'value': stoch_k, 'signal': 'Neutral', 'weight': 0.5})
    
    # Price Position
    if price_pos > 80:
        categories['Bearish'].append(f"Price Position: {price_pos:.0f}% (Overextended)")
        signals.append({'indicator': 'Price Position', 'value': price_pos, 'signal': 'Bearish', 'weight': 1})
    elif price_pos < 20:
        categories['Bullish'].append(f"Price Position: {price_pos:.0f}% (Oversold)")
        signals.append({'indicator': 'Price Position', 'value': price_pos, 'signal': 'Bullish', 'weight': 1})
    else:
        categories['Neutral'].append(f"Price Position: {price_pos:.0f}% (Middle)")
        signals.append({'indicator': 'Price Position', 'value': price_pos, 'signal': 'Neutral', 'weight': 0.5})
    
    # Trend
    if vs_ma20 > 5:
        categories['Bullish'].append(f"Trend: {vs_ma20:.1f}% (Uptrend)")
        signals.append({'indicator': 'Trend', 'value': vs_ma20, 'signal': 'Bullish', 'weight': 2})
    elif vs_ma20 < -5:
        categories['Bearish'].append(f"Trend: {vs_ma20:.1f}% (Downtrend)")
        signals.append({'indicator': 'Trend', 'value': vs_ma20, 'signal': 'Bearish', 'weight': 2})
    else:
        categories['Neutral'].append(f"Trend: {vs_ma20:.1f}% (Sideways)")
        signals.append({'indicator': 'Trend', 'value': vs_ma20, 'signal': 'Neutral', 'weight': 0.5})
    
    return categories, signals

# ---------------------------
# 4. ML MODEL
# ---------------------------
def train_model(data):
    """Train ML model"""
    features = [
        'ret_1', 'ret_3', 'ret_5', 'rsi', 'macd_hist',
        'vs_ma5', 'vs_ma10', 'bb_pct_b', 'vol_ratio',
        'volatility', 'stoch_k', 'price_position', 'vwap_dev'
    ]
    
    available_features = [f for f in features if f in data.columns]
    X = data[available_features].dropna()
    y = data['target'].loc[X.index]
    
    if len(X) < 50:
        return None, None, None, 0
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, available_features, accuracy

def predict_with_model(model, scaler, features, data):
    if model is None:
        return 50, {}
    
    latest = data[features].iloc[-1:].values
    latest_scaled = scaler.transform(latest)
    
    prob = model.predict_proba(latest_scaled)[0][1] * 100
    
    if hasattr(model, 'feature_importances_'):
        importance = dict(zip(features, model.feature_importances_))
    else:
        importance = {}
    
    return prob, importance

# ---------------------------
# 5. VERDICT GENERATION
# ---------------------------
def generate_verdict(signals, ml_prob, price, atr_pct, importance):
    """Generate final verdict"""
    
    total_score = 0
    total_weight = 0
    
    for signal in signals:
        if signal['signal'] == 'Bullish':
            total_score += signal['weight']
        elif signal['signal'] == 'Bearish':
            total_score -= signal['weight']
        total_weight += signal['weight']
    
    indicator_score = total_score / total_weight if total_weight > 0 else 0
    ml_score = (ml_prob - 50) / 50
    combined_score = (indicator_score * 0.55) + (ml_score * 0.45)
    
    # Top features
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
    top_names = [f"{name}" for name, val in top_features if val > 0.05]
    
    if combined_score > 0.3:
        verdict = "BUY 🟢"
        color = "buy-signal"
        reasoning = f"Strong bullish signals. Key factors: {', '.join(top_names) if top_names else 'Multiple indicators align'}"
        confidence = min(100, (combined_score + 1) * 50)
    elif combined_score > 0.1:
        verdict = "CAUTIOUS BUY 🟡"
        color = "wait-signal"
        reasoning = f"Mildly bullish. Key factors: {', '.join(top_names) if top_names else 'Mixed signals'}"
        confidence = min(100, (combined_score + 1) * 40)
    elif combined_score > -0.1:
        verdict = "WAIT 🟡"
        color = "wait-signal"
        reasoning = "Mixed signals, no clear direction"
        confidence = 50
    elif combined_score > -0.3:
        verdict = "CAUTIOUS AVOID 🟠"
        color = "wait-signal"
        reasoning = f"Mildly bearish. Key factors: {', '.join(top_names) if top_names else 'Mixed signals'}"
        confidence = min(100, (1 - combined_score) * 40)
    else:
        verdict = "AVOID 🔴"
        color = "avoid-signal"
        reasoning = f"Strong bearish signals. Key factors: {', '.join(top_names) if top_names else 'Multiple indicators align'}"
        confidence = min(100, (1 - combined_score) * 50)
    
    expected_move = atr_pct * 1.5
    expected_up = price * (1 + expected_move/100)
    expected_down = price * (1 - expected_move/100)
    
    return {
        'verdict': verdict,
        'color': color,
        'reasoning': reasoning,
        'confidence': confidence,
        'combined_score': combined_score,
        'indicator_score': indicator_score,
        'ml_score': ml_score,
        'expected_up': expected_up,
        'expected_down': expected_down,
        'expected_move': expected_move
    }

# ---------------------------
# 6. CHART
# ---------------------------
def create_chart(df, symbol):
    """Create chart with VWAP"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.3, 0.2],
        subplot_titles=(f'{symbol} - Price with VWAP', 'RSI', 'Volume')
    )
    
    # Price with VWAP
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['vwap'], name='VWAP', 
                   line=dict(color='cyan', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['ma_5'], name='MA5', line=dict(color='orange', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['ma_20'], name='MA20', line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    # RSI - FIXED LINE
    fig.add_trace(
        go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volume
    colors = ['green' if c > o else 'red' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        template='plotly_dark',
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    
    return fig

# ---------------------------
# 7. MAIN APP
# ---------------------------
def main():
    st.markdown('<div class="main-header">🇮🇳 Intraday Stock Predictor - All Indian Stocks</div>', unsafe_allow_html=True)
    
    stock_list = get_comprehensive_stock_list()
    
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        search_query = st.text_input("🔍 Search Stock", placeholder="Type symbol...")
        
        if search_query:
            filtered_stocks = search_stocks(search_query, stock_list)
            stock_symbol = st.selectbox("Select Stock", filtered_stocks)
        else:
            popular = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 
                      'ITC', 'HINDUNILVR', 'SBIN', 'POLYCAB', 'BOSCHLTD',
                      'HAL', 'DIXON', 'LLOYDSENGG']
            stock_symbol = st.selectbox("Select Stock (or search above)", popular)
        
        timeframe = st.selectbox("Timeframe", ['15m', '30m', '1h', '1d'], index=1)
        period = st.selectbox("Data Period", ['5d', '10d', '1mo', '3mo'], index=1)
        
        st.markdown("---")
        st.markdown(f"<div class='stocks-count'>📈 {len(stock_list)} stocks available</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 📋 How to Use")
        st.markdown("""
        1. Search for any stock
        2. Select timeframe
        3. Click 'Predict'
        4. Review indicators
        5. Check final verdict
        """)
        
        predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner(f"Analyzing {stock_symbol}..."):
            try:
                ticker = yf.Ticker(f"{stock_symbol}.NS")
                df = ticker.history(period=period, interval=timeframe)
                
                if df.empty:
                    st.error(f"No data found for {stock_symbol}")
                    return
                
                df = calculate_indicators(df)
                df = df.dropna()
                
                if len(df) < 30:
                    st.error(f"Insufficient data for {stock_symbol}")
                    return
                
                latest_price = df['Close'].iloc[-1]
                model, scaler, features, accuracy = train_model(df)
                ml_prob, importance = predict_with_model(model, scaler, features, df)
                categories, signals = categorize_indicators(df)
                
                atr_pct = df['atr_pct'].iloc[-1]
                verdict = generate_verdict(signals, ml_prob, latest_price, atr_pct, importance)
                
                # Display verdict
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div class="{verdict['color']}">
                        {verdict['verdict']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Current Price", f"₹{latest_price:,.2f}")
                
                with col2:
                    st.metric("ML Probability", f"{ml_prob:.1f}%", 
                              delta=f"{ml_prob-50:.1f}%")
                
                with col3:
                    st.metric("Confidence", f"{verdict['confidence']:.0f}%")
                
                with col4:
                    st.metric("Expected UP", f"₹{verdict['expected_up']:,.0f}")
                
                with col5:
                    st.metric("Expected DOWN", f"₹{verdict['expected_down']:,.0f}")
                
                if accuracy > 0:
                    st.info(f"🎯 Model Accuracy: {accuracy*100:.1f}% on test data")
                
                # Indicator categorization
                st.markdown("## 📊 Indicator Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🟢 Bullish Indicators")
                    if categories['Bullish']:
                        for item in categories['Bullish']:
                            st.markdown(f"✅ {item}")
                    else:
                        st.markdown("*No bullish signals*")
                
                with col2:
                    st.markdown("### 🟡 Neutral Indicators")
                    if categories['Neutral']:
                        for item in categories['Neutral']:
                            st.markdown(f"➖ {item}")
                    else:
                        st.markdown("*No neutral signals*")
                
                with col3:
                    st.markdown("### 🔴 Bearish Indicators")
                    if categories['Bearish']:
                        for item in categories['Bearish']:
                            st.markdown(f"❌ {item}")
                    else:
                        st.markdown("*No bearish signals*")
                
                # Reasoning
                st.markdown("## 💡 Reasoning")
                st.info(f"""
                **Verdict:** {verdict['verdict']}
                
                **Reasoning:** {verdict['reasoning']}
                
                **Combined Score:** {verdict['combined_score']:.2f}
                - Indicator Score: {verdict['indicator_score']:.2f}
                - ML Score: {verdict['ml_score']:.2f}
                
                **Expected Move:** ±{verdict['expected_move']:.2f}%
                """)
                
                # Chart
                st.markdown("## 📈 Price Chart with VWAP")
                fig = create_chart(df, stock_symbol)
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("📊 View Raw Data"):
                    st.dataframe(df.tail(20))
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    else:
        st.info("👈 Search for a stock and click 'Predict' to start")
        st.markdown("""
        ### 📊 Features
        - **All Indian Stocks** - 200+ major stocks available
        - **VWAP** - Volume Weighted Average Price for institutional levels
        - **12+ Technical Indicators** - RSI, MACD, MA, Bollinger, Volume, Stochastic, Price Position, Trend
        - **ML-powered prediction** - Random Forest model
        - **Clear verdict** - BUY, WAIT, or AVOID with confidence
        - **Interactive charts** - Candlestick with VWAP and indicators
        """)

# ---------------------------
# 8. RUN APP
# ---------------------------
if __name__ == "__main__":
    main()