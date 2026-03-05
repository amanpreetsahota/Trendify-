import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import yfinance as yf
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Custom modules
from db_manager import get_users, add_user, get_portfolio, add_portfolio_entry, update_portfolio_entry, delete_portfolio_entry
from analysis import show_analysis
from fundamentals import show_fundamentals
from recommendation import generate_recommendation
from portfolio import show_portfolio
from prediction import show_price_prediction

#  PAGE CONFIG 
st.set_page_config(page_title="Trendify – Track Trends. Predict Smarter.",layout="wide",initial_sidebar_state="expanded")

#  CSS 
st.markdown("""
<style>
html, body, [class*="css"] {font-family: 'Inter', sans-serif; background-color: #F8FAFC;}
.main .block-container {padding-top: 2rem; max-width: 1100px;}
.balance-header {background: linear-gradient(135deg, #0061FF 0%, #60EFFF 100%); color:white; padding:40px; border-radius:35px; margin-bottom:30px; box-shadow: 0 10px 20px rgba(0,97,255,0.2);}
.stock-card {background:white; padding:24px; border-radius:24px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); margin-bottom:20px; border:1px solid #F1F5F9;}
.news-row {display:flex; align-items:center; padding:15px; background:white; border-radius:20px; margin-bottom:12px; border:1px solid #F1F5F9; transition:0.2s;}
.news-row:hover {transform: scale(1.01); border-color:#0061FF;}
.rec-pill {padding:8px 20px; border-radius:50px; font-weight:700; display:inline-block; color:white; text-transform:uppercase; font-size:14px;}
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data_features")
MODEL_PATH = os.path.join(BASE_DIR, "models")

# ================= SESSION STATE =================
if "users" not in st.session_state:
    st.session_state.users = get_users()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "learning_mode" not in st.session_state:
    st.session_state.learning_mode = False


# ================= AUTH UI =================
def login_signup_ui():

    st.markdown("<h1 style='text-align:center;'>Trendify</h1>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Signup"])

    # -------- LOGIN --------
    with tab1:
        login_user = st.text_input("Username")
        login_pass = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):

            users = st.session_state.users

            if login_user in users and users[login_user][1] == login_pass:

                st.session_state.logged_in = True
                st.session_state.username = login_user
                st.session_state.user_id = users[login_user][0]

                st.success("Login successful")
                st.rerun()

            else:
                st.error("Invalid username or password")

    # -------- SIGNUP --------
    with tab2:

        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")

        if st.button("Create Account", use_container_width=True):

            users = st.session_state.users

            if new_user in users:
                st.error("Username already exists")

            elif new_user == "" or new_pass == "":
                st.warning("Please fill all fields")

            else:
                add_user(new_user, new_pass)
                st.session_state.users = get_users()

                st.success("Account created! Please login.")


# Show login if not logged in
if not st.session_state.logged_in:
    login_signup_ui()
    st.stop()


# ================= SIDEBAR =================
with st.sidebar:

    st.write(f"Welcome **{st.session_state.username}**")

    if st.button("Logout"):

        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.user_id = None

        st.rerun()
# ================= STOCK SELECTION =================

stocks = {
    "TCS": "TCS.csv",
    "INFY": "INFY.csv",
    "RELIANCE": "RELIANCE.csv",
    "HDFCBANK": "HDFCBANK.csv",
    "ICICIBANK": "ICICIBANK.csv"
}

stock_name = st.sidebar.selectbox("Select Stock", list(stocks.keys()))

use_live = st.sidebar.toggle("Use Live Market Data", value=True)

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
    return df.dropna()

def calculate_indicators(df):
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

df = get_processed_data(stock_name, stocks[stock_name], use_live)
df = calculate_indicators(df)
latest = df.iloc[-1]

# ================= FUNDAMENTALS =================
ticker = yf.Ticker(stock_name + ".NS")
info = ticker.info

# ================= ML PREDICTION =================
FEATURES = ["open","high","low","close","volume","daily_return","sma_20","sma_50"]
reg_model_path = os.path.join(MODEL_PATH, stocks[stock_name].replace(".NS.csv","_rf_regression.pkl"))
reg_model = joblib.load(reg_model_path)

X = latest[FEATURES].values.reshape(1,-1)
try:
    pred_price = reg_model.predict(X)[0]
except Exception:
    pred_price = latest["close"]

# ================= TOP HEADER =================
change_pct = ((pred_price - latest["close"])/latest["close"])*100
st.markdown(f"""
<div class="balance-header">
<p style="margin:0; opacity:0.9; font-size:16px; font-weight:500;">Predicted Next Close • {stock_name}</p>
<h1 style="margin:5px 0; font-size:48px; font-weight:700;">₹ {pred_price:.2f}</h1>
<div style="background:rgba(255,255,255,0.2); padding:6px 16px; border-radius:50px; display:inline-block; font-size:14px;">
{'▲' if change_pct>0 else '▼'} {abs(change_pct):.2f}% Expected Move
</div>
<p style="margin-top:5px; font-size:16px;">Current Price: ₹ {latest['close']:.2f}</p>
</div>
""", unsafe_allow_html=True)

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs(["Portfolio","Stock Analysis","Fundamentals","News"])

with tab1:
    show_portfolio()

with tab2:
    st.subheader(f"📈 Price Chart with SMA & RSI • {stock_name}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], increasing_line_color="#0061FF", decreasing_line_color="#FF4B4B", name="Price"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["sma_20"], mode="lines", line=dict(color="#FFA500"), name="SMA 20"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["sma_50"], mode="lines", line=dict(color="#00CFFF"), name="SMA 50"))
    fig.update_layout(height=400, xaxis_rangeslider_visible=False, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📊 RSI (14)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df["date"], y=df["rsi"], mode="lines", line=dict(color="#6F42C1"), name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="top right")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", annotation_position="bottom right")
    fig_rsi.update_layout(height=200, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    st.subheader("🔮 5–7 Day Price Forecast")
    future_prices = [latest["close"]]
    last_row = latest.copy()
    for _ in range(7):
        X_future = last_row[FEATURES].values.reshape(1,-1)
        try:
            next_price = reg_model.predict(X_future)[0]
        except:
            next_price = last_row["close"]
        future_prices.append(next_price)
        # Update rolling features
        last_row["open"] = last_row["close"]
        last_row["high"] = next_price*1.01
        last_row["low"] = next_price*0.99
        last_row["close"] = next_price
        last_row["sma_20"] = (last_row["sma_20"]*19 + next_price)/20
        last_row["sma_50"] = (last_row["sma_50"]*49 + next_price)/50
    future_dates = pd.date_range(df["date"].iloc[-1]+pd.Timedelta(days=1), periods=7)
    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(x=future_dates, y=future_prices[1:], mode="lines+markers", name="Predicted Price", line=dict(color="#22C55E")))
    fig_future.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_future, use_container_width=True)
    
with tab3:
    show_fundamentals(stock_name, info, latest, st.session_state.learning_mode)

with tab4:
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