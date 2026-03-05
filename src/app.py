import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
import yfinance as yf
import requests
from datetime import datetime, timedelta

# Custom modules
from db_manager import get_users, add_user, get_portfolio, add_portfolio_entry, update_portfolio_entry, delete_portfolio_entry
from analysis import show_analysis
from fundamentals import show_fundamentals
from prediction import show_price_prediction
from recommendation import generate_recommendation
from portfolio import show_portfolio

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Trendify – Track Trends. Predict Smarter.",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= FANCY FINTECH CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F8FAFC;
}

/* Main Container Padding */
.main .block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

/* Blue Gradient Header Card */
.balance-header {
    background: linear-gradient(135deg, #0061FF 0%, #60EFFF 100%);
    color: white;
    padding: 40px;
    border-radius: 35px;
    margin-bottom: 30px;
    box-shadow: 0 10px 20px rgba(0, 97, 255, 0.2);
}

/* White Content Cards */
.stock-card {
    background-color: white;
    padding: 24px;
    border-radius: 24px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    margin-bottom: 20px;
    border: 1px solid #F1F5F9;
}

/* News Row Style */
.news-row {
    display: flex; 
    align-items: center; 
    padding: 15px; 
    background: white; 
    border-radius: 20px; 
    margin-bottom: 12px; 
    border: 1px solid #F1F5F9;
    transition: 0.2s;
}
.news-row:hover {
    transform: scale(1.01);
    border-color: #0061FF;
}

/* Custom Recommendation Pill */
.rec-pill {
    padding: 8px 20px;
    border-radius: 50px;
    font-weight: 700;
    display: inline-block;
    color: white;
    text-transform: uppercase;
    font-size: 14px;
}

/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ================= PATHS & DATA =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data_features")
MODEL_PATH = os.path.join(BASE_DIR, "models")

# ================= SESSION STATE =================
if "users" not in st.session_state: st.session_state.users = get_users()
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = ""
if "user_id" not in st.session_state: st.session_state.user_id = None
if "learning_mode" not in st.session_state: st.session_state.learning_mode = False

# ================= AUTH UI =================
def login_signup_ui():
    st.markdown("<h1 style='text-align:center; color:#1E293B;'>Trendify</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Login", "Signup"])
    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Sign In", use_container_width=True):
            users = st.session_state.users
            if u in users and users[u][1] == p:
                st.session_state.logged_in = True
                st.session_state.username = u
                st.session_state.user_id = users[u][0]
                st.rerun()
            else: st.error("Invalid credentials")
if not st.session_state.logged_in:
    login_signup_ui()
    st.stop()

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown(f"### Hello, **{st.session_state.username}** 👋")
    stocks = {"TCS": "TCS.NS.csv", "INFY": "INFY.NS.csv", "RELIANCE": "RELIANCE.NS.csv", "HDFCBANK": "HDFCBANK.NS.csv"}
    stock_name = st.selectbox("Select Asset", list(stocks.keys()))
    st.session_state.learning_mode = st.toggle("🎓 Learning Mode", st.session_state.learning_mode)
    use_live = st.toggle("🔴 Live Market", False)
    if st.button("Log Out", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# ================= DATA FETCHING =================
@st.cache_data(ttl=300)
def get_processed_data(symbol, file_name, live=False):
    if live:
        df = yf.download(symbol + ".NS", period="6mo", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.columns = [col.lower() for col in df.columns]
    else:
        df = pd.read_csv(os.path.join(DATA_PATH, file_name))
        df["date"] = pd.to_datetime(df["date"])
    
    df["daily_return"] = df["close"].pct_change()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    
    # RSI calculation
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df.dropna()

# ================= INDICATORS =================
def calculate_indicators(df):
    # SMA
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df

# ================= INDICATORS =================
def calculate_indicators(df):
    # SMA
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df

df = calculate_indicators(df)
latest = df.iloc[-1]

# ================= PREDICTION =================
# Features used in training (must match your trained model)
FEATURES = ["open","high","low","close","volume","daily_return","sma_10","sma_50"]

X = latest[FEATURES].values.reshape(1,-1)
reg_model = joblib.load(os.path.join(MODEL_PATH, stocks[stock_name].replace(".csv", "_rf_regression.pkl")))

try:
    pred_price = reg_model.predict(X)[0]
except ValueError as e:
    st.error(f"Prediction failed: {e}")
    pred_price = latest["close"]  # fallback to last close

# ================= 5–7 DAY FUTURE PREDICTION =================
future_prices = [latest["close"]]
for _ in range(6):  # next 6 days
    X_future = df[FEATURES].iloc[-1].values.reshape(1,-1)
    next_price = reg_model.predict(X_future)[0]
    future_prices.append(next_price)
    # append to df for rolling features if needed
    temp = df.iloc[-1].copy()
    temp["close"] = next_price
    df = pd.concat([df, pd.DataFrame([temp])], ignore_index=True)

# ================= DASHBOARD CHARTS =================
import plotly.graph_objects as go
st.subheader("📈 Price Chart with SMA & RSI")

fig = go.Figure()
# Candlestick
fig.add_trace(go.Candlestick(
    x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    name="Price", increasing_line_color="#0061FF", decreasing_line_color="#FF4B4B"
))
# SMA lines
fig.add_trace(go.Scatter(x=df["date"], y=df["sma_20"], mode="lines", line=dict(color="#FFA500"), name="SMA 20"))
fig.add_trace(go.Scatter(x=df["date"], y=df["sma_50"], mode="lines", line=dict(color="#00CFFF"), name="SMA 50"))
fig.update_layout(height=400, xaxis_rangeslider_visible=False, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)

# RSI Chart
st.subheader("📊 RSI (14)")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df["date"], y=df["rsi_14"], mode="lines", line=dict(color="#6F42C1"), name="RSI 14"))
fig_rsi.update_layout(height=200, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_rsi, use_container_width=True)

# ================= USER LEARNING =================
st.subheader("🎓 Learning Mode: Metrics Explanation")
st.markdown(f"""
- **Current Price:** ₹ {latest['close']:.2f}  
- **Predicted Next Close:** ₹ {pred_price:.2f}  
- **Expected % Change:** {((pred_price - latest['close'])/latest['close']*100):.2f}%  

**Indicators:**
- **SMA 20:** Average price over last 20 days.  
- **SMA 50:** Average price over last 50 days.  
- **RSI 14:** Measures overbought (>70) / oversold (<30) conditions.  
""")

# ================= 5–7 DAY FORECAST =================
st.subheader("🔮 5–7 Day Price Forecast")
for i, price in enumerate(future_prices[1:], 1):
    st.write(f"Day {i}: ₹ {price:.2f}")

# ================= DASHBOARD UI =================

# Header
change_pct = ((pred_price - latest['close']) / latest['close']) * 100
st.markdown(f"""
<div class="balance-header">
<p style="margin:0; opacity:0.9; font-size:16px; font-weight:500;">Predicted Next Close • {stock_name}</p>
<h1 style="margin:5px 0; font-size:48px; font-weight:700;">₹ {pred_price:.2f}</h1>
<div style="background:rgba(255,255,255,0.2); padding:6px 16px; border-radius:50px; display:inline-block; font-size:14px;">
{'▲' if change_pct > 0 else '▼'} {abs(change_pct):.2f}% Expected Move
</div>
</div>
""", unsafe_allow_html=True)

# Price + SMA 20/50 Candlestick Chart
st.markdown('<div class="stock-card">', unsafe_allow_html=True)
st.subheader("Price & SMA Analysis")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                             increasing_line_color='#0061FF', decreasing_line_color='#FF4B4B', name="Price"))
fig.add_trace(go.Scatter(x=df["date"], y=df["sma_20"], mode='lines', name='SMA 20', line=dict(color='orange', width=2)))
fig.add_trace(go.Scatter(x=df["date"], y=df["sma_50"], mode='lines', name='SMA 50', line=dict(color='green', width=2)))
fig.update_layout(plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=True, gridcolor="#F1F5F9"), margin=dict(l=0,r=0,t=0,b=0), height=400,
                  xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# RSI Chart
st.markdown('<div class="stock-card">', unsafe_allow_html=True)
st.subheader("RSI Indicator (14 days)")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df["date"], y=df["rsi"], mode='lines+markers', name='RSI', line=dict(color='#FF9900')))
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="top right")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", annotation_position="bottom right")
fig_rsi.update_layout(plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False),
                      yaxis=dict(showgrid=True, gridcolor="#F1F5F9"), margin=dict(l=0,r=0,t=0,b=0), height=250)
st.plotly_chart(fig_rsi, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# 5–7 days prediction
st.markdown('<div class="stock-card">', unsafe_allow_html=True)
st.subheader("Next 5–7 Days Predicted Prices")
future_days = 7
last_row = latest.copy()
future_preds = []
for i in range(future_days):
    X_future = last_row[["open","high","low","close","volume","daily_return","sma_10","sma_20","sma_50"]].values.reshape(1,-1)
    pred = reg_model.predict(X_future)[0]
    future_preds.append(pred)
    last_row["open"] = last_row["close"]
    last_row["high"] = pred*1.01
    last_row["low"] = pred*0.99
    last_row["close"] = pred
    last_row["daily_return"] = 0
    last_row["sma_10"] = (last_row["sma_10"]*9 + pred)/10
    last_row["sma_20"] = (last_row["sma_20"]*19 + pred)/20
    last_row["sma_50"] = (last_row["sma_50"]*49 + pred)/50
future_dates = pd.date_range(df["date"].iloc[-1]+pd.Timedelta(days=1), periods=future_days)
fig_future = go.Figure()
fig_future.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines+markers', name='Predicted Price', line=dict(color='#22C55E')))
fig_future.update_layout(plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False),
                         yaxis=dict(showgrid=True, gridcolor="#F1F5F9"), margin=dict(l=0,r=0,t=0,b=0), height=300)
st.plotly_chart(fig_future, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Fundamentals + recommendation
col_a, col_b = st.columns([1,1])
with col_a:
    show_fundamentals(stock_name, info, latest, st.session_state.learning_mode)
with col_b:
    st.markdown('<div class="stock-card" style="text-align:center;">', unsafe_allow_html=True)
    st.markdown("### AI Signal")
    bg = "#22C55E" if recommendation=="BUY" else "#EF4444"
    st.markdown(f'<div class="rec-pill" style="background:{bg};">{recommendation}</div>', unsafe_allow_html=True)
    if st.session_state.learning_mode:
        st.markdown(f"<p style='margin-top:15px; color:#64748B;'>{reasons[0] if reasons else ''}</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# News
st.subheader("Latest News")
def get_filtered_financial_news(stock_name):
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key: return []
    url = f"https://finnhub.io/api/v1/news?category=general&token={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if not isinstance(data, list): return []
        filtered_news = [a for a in data if stock_name.lower() in a.get("headline","").lower() or stock_name.lower() in a.get("summary","").lower()]
        return filtered_news[:5] if filtered_news else data[:5]
    except: return []

news = get_filtered_financial_news(stock_name)
for article in news[:3]:
    st.markdown(f"""
    <div class="news-row">
    <div style="background:#EFF6FF; padding:12px; border-radius:15px; margin-right:15px;">📰</div>
    <div style="flex-grow:1;">
        <h4 style="margin:0; font-size:15px; color:#1E293B;">{article.get('headline','')}</h4>
        <p style="margin:0; font-size:12px; color:#64748B;">Source: Market Intel</p>
    </div>
    <a href="{article.get('url','#')}" target="_blank" style="text-decoration:none; color:#0061FF; font-weight:700;">VIEW</a>
    </div>
    """, unsafe_allow_html=True)

# Portfolio
st.markdown("---")
show_portfolio()