import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
import yfinance as yf
import requests
from datetime import datetime, timedelta

from db_manager import get_users, add_user, get_portfolio, add_portfolio_entry
from analysis import show_analysis
from fundamentals import show_fundamentals
from prediction import show_price_prediction
from recommendation import generate_recommendation

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Trendify – Track Trends. Predict Smarter.",
    layout="wide"
)

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data_features")
MODEL_PATH = os.path.join(BASE_DIR, "models")

# ================= USER FUNCTIONS =================
if "users" not in st.session_state:
    st.session_state.users = get_users()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

# ================= AUTH =================
def login_signup_ui():
    st.title("🔐 Login / Signup")
    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            users = st.session_state.users
            if u in users and users[u][1] == p:  # users = {username: (id, password)}
                st.session_state.logged_in = True
                st.session_state.username = u
                st.session_state.user_id = users[u][0]
                st.session_state.portfolio = get_portfolio(users[u][0])
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        nu = st.text_input("New Username")
        npass = st.text_input("New Password", type="password")
        if st.button("Signup"):
            users = st.session_state.users
            if nu in users:
                st.warning("User already exists")
            else:
                # Assign new user ID
                new_id = max([uid for uid, pw in [v for v in users.values()]] + [0]) + 1
                add_user(new_id, nu, npass)
                st.session_state.users = get_users()
                st.success("Signup successful")

if not st.session_state.logged_in:
    login_signup_ui()
    st.stop()

# ================= SIDEBAR =================
st.sidebar.success(f"Welcome {st.session_state.username}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

stocks = {
    "TCS": "TCS.NS.csv",
    "INFY": "INFY.NS.csv",
    "RELIANCE": "RELIANCE.NS.csv",
    "HDFCBANK": "HDFCBANK.NS.csv",
    "ICICIBANK": "ICICIBANK.NS.csv"
}

stock_name = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
file_name = stocks[stock_name]
use_live = st.sidebar.toggle("🔴 Use Live Market Data", False)
learning_mode = st.sidebar.selectbox(
    "🎓 Learning Mode",
    ["Off", "Beginner", "Intermediate", "Advanced"]
)

# ================= HELPER FUNCTIONS =================
def calculate_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def get_live_data(symbol):
    df = yf.download(symbol, period="3mo", interval="1d", progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "date"}, inplace=True)
    df.columns = [col.lower() for col in df.columns]
    df["daily_return"] = df["close"].pct_change()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df = calculate_rsi(df)
    df.dropna(inplace=True)
    return df

def get_filtered_financial_news(stock_name):
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return []
    url = f"https://finnhub.io/api/v1/news?category=general&token={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if not isinstance(data, list):
            return []
        filtered_news = [
            article for article in data
            if stock_name.lower() in article.get("headline", "").lower()
            or stock_name.lower() in article.get("summary", "").lower()
        ]
        if len(filtered_news) == 0:
            return data[:5]
        return filtered_news[:5]
    except:
        return []

# ================= LOAD DATA =================
if use_live:
    df = get_live_data(stock_name + ".NS")
else:
    df = pd.read_csv(os.path.join(DATA_PATH, file_name))
    df["date"] = pd.to_datetime(df["date"])

df.sort_values("date", inplace=True)

# ================= LOAD MODELS =================
reg_model = joblib.load(os.path.join(
    MODEL_PATH, file_name.replace(".csv", "_rf_regression.pkl")
))
clf_model = joblib.load(os.path.join(
    MODEL_PATH, file_name.replace(".csv", "_rf_classification.pkl")
))

features = ["open", "high", "low", "close", "volume", "daily_return", "sma_10", "sma_50"]
latest = df.iloc[-1]
X = latest[features].values.reshape(1, -1)

pred_price = reg_model.predict(X)[0]
trend = clf_model.predict(X)[0]

trend_text = "UP 📈" if trend == 1 else "DOWN 📉"
trend_color = "green" if trend == 1 else "red"

# ================= HEADER =================
st.markdown("<h1 style='text-align:center;'>📊 Trendify</h1>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align:center; color:{trend_color};'>{trend_text}</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Last Close", f"₹ {latest['close']:.2f}", help="Most recent closing price of the selected stock.")
col2.metric("Predicted Next Close", f"₹ {pred_price:.2f}", help="ML prediction for next close.")
col3.metric("Expected Change", f"₹ {pred_price - latest['close']:.2f}", help="Difference from last close.")

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["📊 Overview", "🔮 Prediction", "📈 Technicals", "📰 News", "💼 Portfolio", "📉 Analysis", "📊 Fundamentals"]
)

# ---------- Overview ----------
with tab1:
    fig = go.Figure(data=[go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )])
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Prediction ----------
with tab2:
    st.success(f"Predicted Next Close: ₹ {pred_price:.2f}")
    st.info(f"Trend Prediction: {trend_text}")
    show_price_prediction(df, reg_model)

    ticker = yf.Ticker(stock_name + ".NS")
    info = ticker.info
    pe_ratio = info.get("trailingPE", None)
    revenue_growth = info.get("revenueGrowth", 0)
    profit_growth = info.get("earningsGrowth", 0)
    if revenue_growth: revenue_growth *= 100
    if profit_growth: profit_growth *= 100

    recommendation, reasons = generate_recommendation(
        latest["close"], pred_price, pe_ratio, revenue_growth, profit_growth
    )
    st.subheader("📌 Investment Recommendation")
    st.success(f"Recommendation: {recommendation}")
    st.write("### 🔎 Why?")
    for r in reasons:
        st.write(f"- {r}")

    if learning_mode != "Off":
        with st.expander("📘 Why this Recommendation?"):
            if learning_mode == "Beginner":
                st.write("• Price expected direction\n• P/E ratio\n• Company growth")
            elif learning_mode == "Intermediate":
                st.write("Multi-factor scoring: regression, valuation, growth")
            else:
                st.write("ML prediction + fundamental filters + scoring + rules")

# ---------- Technicals ----------
with tab3:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close"))
    fig2.add_trace(go.Scatter(x=df["date"], y=df["sma_10"], name="SMA 10"))
    fig2.add_trace(go.Scatter(x=df["date"], y=df["sma_50"], name="SMA 50"))
    st.plotly_chart(fig2, use_container_width=True)

    if "rsi" in df.columns:
        st.subheader("RSI Indicator")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df["date"], y=df["rsi"], name="RSI", line=dict(color="purple")))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        st.plotly_chart(fig_rsi, use_container_width=True)
        latest_rsi = df["rsi"].iloc[-1]
        if latest_rsi > 70:
            st.error(f"RSI: {latest_rsi:.2f} → Overbought")
        elif latest_rsi < 30:
            st.success(f"RSI: {latest_rsi:.2f} → Oversold")
        else:
            st.info(f"RSI: {latest_rsi:.2f} → Neutral")
    else:
        st.warning("RSI data not available.")

# ---------- News ----------
with tab4:
    st.subheader("📰 Latest Financial News")
    news = get_filtered_financial_news(stock_name)
    if news:
        for article in news:
            st.markdown(f"### {article.get('headline','No Title')}")
            st.write(f"Source: {article.get('source','Unknown')}")
            st.write(article.get("summary",""))
            if article.get("url"): st.markdown(f"[Read more]({article['url']})")
            st.write("---")
    else:
        st.info("Unable to fetch news.")

# ---------- Portfolio ----------
with tab5:
    from portfolio import show_portfolio
    show_portfolio()

# ---------- Analysis ----------
with tab6:
    show_analysis(stock_name + ".NS")

# ---------- Fundamentals ----------
with tab7:
    show_fundamentals(stock_name + ".NS")
    if learning_mode != "Off":
        with st.expander("📘 Understanding Fundamental Metrics"):
            st.write("• P/E Ratio → Expensive or cheap\n• Revenue Growth → Sales growth\n• Profit Growth → Earnings growth")

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
### ⚠ Disclaimer  
Trendify provides AI-based analytical insights for educational purposes only.  
This platform does not provide financial advice.  
Investments in stock markets are subject to market risk.
""")