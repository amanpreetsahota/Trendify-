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

# ================= INDICATORS FUNCTIONS =================
def calculate_indicators(df):
    # SMA
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df

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
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #F8FAFC; }
.main .block-container { padding-top: 2rem; max-width: 1100px; }
.balance-header { background: linear-gradient(135deg, #0061FF 0%, #60EFFF 100%); color: white; padding: 40px; border-radius: 35px; margin-bottom: 30px; box-shadow: 0 10px 20px rgba(0, 97, 255, 0.2); }
.stock-card { background-color: white; padding: 24px; border-radius: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); margin-bottom: 20px; border: 1px solid #F1F5F9; }
.news-row { display: flex; align-items: center; padding: 15px; background: white; border-radius: 20px; margin-bottom: 12px; border: 1px solid #F1F5F9; transition: 0.2s; }
.news-row:hover { transform: scale(1.01); border-color: #0061FF; }
.rec-pill { padding: 8px 20px; border-radius: 50px; font-weight: 700; display: inline-block; color: white; text-transform: uppercase; font-size: 14px; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
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
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.columns = [col.lower() for col in df.columns]
    else:
        df = pd.read_csv(os.path.join(DATA_PATH, file_name))
        df["date"] = pd.to_datetime(df["date"])
    df["daily_return"] = df["close"].pct_change()
    return df.dropna()

df = get_processed_data(stock_name, stocks[stock_name], use_live)
df = calculate_indicators(df)
latest = df.iloc[-1]

# ================= PREDICTION =================
FEATURES = ["open","high","low","close","volume","daily_return","sma_20","sma_50"]
reg_model = joblib.load(os.path.join(MODEL_PATH, stocks[stock_name].replace(".csv", "_rf_regression.pkl")))

try:
    X = latest[FEATURES].values.reshape(1,-1)
    pred_price = reg_model.predict(X)[0]
except ValueError:
    pred_price = latest["close"]

# ================= TABS =================
tab_dashboard, tab_portfolio, tab_news = st.tabs(["📈 Dashboard", "💼 Portfolio", "📰 News"])

# ------------------ DASHBOARD ------------------
with tab_dashboard:
    # Header
    change_pct = ((pred_price - latest['close']) / latest['close']) * 100
    st.markdown(f"""
    <div class="balance-header">
        <p style="margin:0; opacity:0.9; font-size:16px; font-weight:500;">{stock_name} • Predicted Next Close</p>
        <h1 style="margin:5px 0; font-size:48px; font-weight:700;">₹ {pred_price:.2f}</h1>
        <div style="background:rgba(255,255,255,0.2); padding:6px 16px; border-radius:50px; display:inline-block; font-size:14px;">
            {'▲' if change_pct > 0 else '▼'} {abs(change_pct):.2f}% Expected Move
        </div>
        <p style="margin:5px 0; font-size:16px; opacity:0.7;">Current Price: ₹ {latest['close']:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    # Price Chart + SMA
    st.subheader("Price & Indicators")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price", increasing_line_color="#0061FF", decreasing_line_color="#FF4B4B"
    ))
    fig.add_trace(go.Scatter(x=df["date"], y=df["sma_20"], mode="lines", line=dict(color="#FFA500"), name="SMA 20"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["sma_50"], mode="lines", line=dict(color="#00CFFF"), name="SMA 50"))
    st.plotly_chart(fig, use_container_width=True)

    # RSI Chart
    st.subheader("RSI (14)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df["date"], y=df["rsi_14"], mode="lines", line=dict(color="#6F42C1"), name="RSI 14"))
    st.plotly_chart(fig_rsi, use_container_width=True)

    # Fundamentals + Recommendation
    info = {}  # ensure info is available to prevent TypeError
    col_a, col_b = st.columns([1,1])
    with col_a:
        show_fundamentals(stock_name, info, latest, st.session_state.learning_mode)
    with col_b:
        st.markdown('<div class="stock-card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown("### AI Signal")
        recommendation = generate_recommendation(latest)
        bg = "#22C55E" if recommendation=="BUY" else "#EF4444"
        st.markdown(f'<div class="rec-pill" style="background:{bg};">{recommendation}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------ PORTFOLIO ------------------
with tab_portfolio:
    show_portfolio()

# ------------------ NEWS ------------------
with tab_news:
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
    for article in news[:5]:
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