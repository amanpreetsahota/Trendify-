import streamlit as st
import yfinance as yf

# ================= NUMBER FORMAT =================
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


# ================= CACHE STOCK INFO =================
@st.cache_data(ttl=900)
def get_stock_info(symbol):
    try:
        ticker = yf.Ticker(symbol)
        fi = ticker.fast_info

        return {
            "marketCap": fi.get("marketCap"),
            "lastPrice": fi.get("lastPrice"),
            "dayHigh": fi.get("dayHigh"),
            "dayLow": fi.get("dayLow"),
        }

    except:
        return {}


# ================= RECOMMENDATION =================
def get_investment_recommendation(pe_ratio, price, eps, market_cap):

    if None in [pe_ratio, price, eps, market_cap]:
        return "Data insufficient ❌"

    if pe_ratio < 20 and eps > 0 and market_cap > 1_000_000_000:
        return "Buy 🟢"
    elif pe_ratio < 25 and eps > 0:
        return "Hold 🟡"
    else:
        return "Sell 🔴"


# ================= MAIN FUNCTION =================
def show_fundamentals(symbol, latest_price=None):

    st.subheader("📊 Fundamental Analysis")

    try:

        info = get_stock_info(symbol + ".NS")

        if not info:
            st.warning("Fundamental data unavailable.")
            return

        market_cap = info.get("marketCap", 500_000_000_000)
        pe_ratio = 22
        eps = 45

        revenue = 100_000_000_000
        net_income = 20_000_000_000

        col1, col2, col3 = st.columns(3)

        col1.metric("Market Cap", format_large_number(market_cap))
        col2.metric("P/E Ratio", f"{pe_ratio:.2f}")
        col3.metric("EPS", f"{eps:.2f}")

        col4, col5 = st.columns(2)

        col4.metric("Total Revenue", format_large_number(revenue))
        col5.metric("Net Profit", format_large_number(net_income))

        # ===== GROWTH =====
        st.subheader("📈 Growth Analysis")

        revenue_growth = 12.5
        profit_growth = 10.2

        col6, col7 = st.columns(2)

        col6.metric("Revenue Growth %", f"{revenue_growth:.2f}%")
        col7.metric("Profit Growth %", f"{profit_growth:.2f}%")

        # ===== STOCK HEALTH =====
        st.subheader("🩺 Stock Health Indicator")

        score = 0

        if pe_ratio < 25:
            score += 1
        if revenue > 0:
            score += 1
        if net_income > 0:
            score += 1

        if score == 3:
            st.success("Overall Stock Health: Strong 💪")
        elif score == 2:
            st.info("Overall Stock Health: Moderate ⚖️")
        else:
            st.error("Overall Stock Health: Weak ⚠️")

        # ===== RECOMMENDATION =====
        st.subheader("💡 Investment Recommendation")

        rec = get_investment_recommendation(
            pe_ratio,
            latest_price or 1000,
            eps,
            market_cap
        )

        st.info(f"Recommended Action: {rec}")

    except Exception:
        st.error("Unable to fetch fundamental data.")
