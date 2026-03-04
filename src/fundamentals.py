import streamlit as st
import yfinance as yf
import pandas as pd

def format_large_number(num):
    if num is None:
        return "N/A"

    try:
        num = float(num)
    except:
        return "N/A"

    if num >= 1_000_000_000_000:
        return f"₹ {num / 1_000_000_000_000:.2f} Trillion"
    elif num >= 1_000_000_000:
        return f"₹ {num / 1_000_000_000:.2f} Billion"
    elif num >= 1_000_000:
        return f"₹ {num / 1_000_000:.2f} Million"
    else:
        return f"₹ {num:,.2f}"

def show_fundamentals(symbol):
    st.subheader("📊 Fundamental Analysis")

    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        # ================= BASIC METRICS =================
        market_cap = info.get("marketCap")
        pe_ratio = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        eps = info.get("trailingEps")
        revenue = info.get("totalRevenue")
        net_income = info.get("netIncomeToCommon")

        col1, col2, col3 = st.columns(3)

        col1.metric("Market Cap", format_large_number(market_cap))
        col2.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
        col3.metric("EPS", f"{eps:.2f}" if eps else "N/A")

        col4, col5 = st.columns(2)
        col4.metric("Total Revenue", format_large_number(revenue))
        col5.metric("Net Profit", format_large_number(net_income))

        # ================= GROWTH CALCULATION =================
        st.subheader("📈 Growth Analysis")

        income_stmt = stock.income_stmt

        if income_stmt is not None and not income_stmt.empty:

            if "Total Revenue" in income_stmt.index:
                revenue_series = income_stmt.loc["Total Revenue"]
            else:
                revenue_series = None

            if "Net Income" in income_stmt.index:
                net_series = income_stmt.loc["Net Income"]
            else:
                net_series = None

            if revenue_series is not None and len(revenue_series) > 1:
                revenue_growth = (
                    (revenue_series.iloc[0] - revenue_series.iloc[1])
                    / revenue_series.iloc[1]
                ) * 100
            else:
                revenue_growth = None

            if net_series is not None and len(net_series) > 1:
                profit_growth = (
                    (net_series.iloc[0] - net_series.iloc[1])
                    / net_series.iloc[1]
                ) * 100
            else:
                profit_growth = None

            col6, col7 = st.columns(2)

            col6.metric(
                "Revenue Growth %",
                f"{revenue_growth:.2f}%" if revenue_growth is not None else "N/A")

            col7.metric(
                 "Profit Growth %",
                f"{profit_growth:.2f}%" if profit_growth is not None else "N/A")

        else:
            st.warning("Growth data not available.")

        # ================= STOCK HEALTH SCORE =================
        st.subheader("🩺 Stock Health Indicator")

        score = 0

        if pe_ratio is not None and pe_ratio < 25:
            score += 1
        if revenue is not None and revenue > 0:
            score += 1
        if net_income is not None and net_income > 0:
            score += 1

        if score == 3:
            health = "Strong 💪"
            st.success(f"Overall Stock Health: {health}")
        elif score == 2:
            health = "Moderate ⚖️"
            st.info(f"Overall Stock Health: {health}")
        else:
            health = "Weak ⚠️"
            st.error(f"Overall Stock Health: {health}")

    except Exception as e:
        st.error("Unable to fetch fundamental data.")