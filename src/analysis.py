import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def show_analysis(symbol):

    st.header("📉 Advanced Historical Analysis")

    # -------------------------
    # Time Range Selector
    # -------------------------
    period = st.selectbox(
        "Select Time Period",
        ["3mo", "6mo", "1y", "2y", "5y"],
        index=2
    )

    # -------------------------
    # Fetch Data
    # -------------------------
    df = yf.download(symbol, period=period, auto_adjust=True)

    if df.empty:
        st.error("No data found.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # -------------------------
    # Moving Averages
    # -------------------------
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # -------------------------
    # RSI Calculation
    # -------------------------
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)

    # -------------------------
    # Trend Summary Section
    # -------------------------
    current_price = df["Close"].iloc[-1]
    ma20 = df["MA20"].iloc[-1]
    ma50 = df["MA50"].iloc[-1]
    rsi = df["RSI"].iloc[-1]

    trend = "Bullish" if ma20 > ma50 else "Bearish"

    if rsi > 70:
        rsi_status = "Overbought"
    elif rsi < 30:
        rsi_status = "Oversold"
    else:
        rsi_status = "Neutral"

    col1, col2, col3 = st.columns(3)

    col1.metric("Trend", trend)
    col2.metric("RSI Status", rsi_status)
    col3.metric("Current Price", f"₹{current_price:.2f}")

    st.markdown("---")

    # -------------------------
    # Create Subplots
    # -------------------------
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2]
    )

    # Row 1: Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price"
        ),
        row=1,
        col=1
    )

    # Moving averages
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MA20"], name="MA20"),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df["MA50"], name="MA50"),
        row=1,
        col=1
    )

    # Row 2: Volume
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume"),
        row=2,
        col=1
    )

    # Row 3: RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df["RSI"], name="RSI"),
        row=3,
        col=1
    )

    fig.add_hline(y=70, line_dash="dash", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", row=3, col=1)

    fig.update_layout(
        height=900,
        template="plotly_dark",
        title=f"{symbol} Advanced Technical Analysis",
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("RSI > 70 = Overbought | RSI < 30 = Oversold")