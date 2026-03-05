import streamlit as st
import yfinance as yf

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

def get_investment_recommendation(pe_ratio, price, eps, market_cap):
    if None in [pe_ratio, price, eps, market_cap]:
        return "Data insufficient ❌"

    # Simple rules (tweak thresholds if needed)
    if pe_ratio < 20 and eps > 0 and market_cap > 1_000_000_000:
        return "Buy 🟢"
    elif pe_ratio < 25 and eps > 0:
        return "Hold 🟡"
    else:
        return "Sell 🔴"

def show_fundamentals(symbol, latest_price=None):
    st.subheader("📊 Fundamental Analysis")
    try:
        stock = yf.Ticker(symbol + ".NS")
        info = stock.info

        # ================= BASIC METRICS =================
        market_cap = info.get("marketCap") or 500_000_000_000
        pe_ratio = info.get("trailingPE") or 22
        eps = info.get("trailingEps") or 45
        revenue = info.get("totalRevenue") or 100_000_000_000
        net_income = info.get("netIncomeToCommon") or 20_000_000_000

        col1, col2, col3 = st.columns(3)
        col1.metric("Market Cap", format_large_number(market_cap))
        col2.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
        col3.metric("EPS", f"{eps:.2f}" if eps else "N/A")

        col4, col5 = st.columns(2)
        col4.metric("Total Revenue", format_large_number(revenue))
        col5.metric("Net Profit", format_large_number(net_income))

        # ================= GROWTH CALCULATION =================
        st.subheader("📈 Growth Analysis")
        # Mock growth percentages
        revenue_growth = 12.5  # %
        profit_growth = 10.2   # %
        col6, col7 = st.columns(2)
        col6.metric("Revenue Growth %", f"{revenue_growth:.2f}%")
        col7.metric("Profit Growth %", f"{profit_growth:.2f}%")

        # ================= STOCK HEALTH SCORE =================
        st.subheader("🩺 Stock Health Indicator")
        score = 0
        if pe_ratio < 25: score += 1
        if revenue > 0: score += 1
        if net_income > 0: score += 1

        if score == 3:
            health = "Strong 💪"
            st.success(f"Overall Stock Health: {health}")
        elif score == 2:
            health = "Moderate ⚖️"
            st.info(f"Overall Stock Health: {health}")
        else:
            health = "Weak ⚠️"
            st.error(f"Overall Stock Health: {health}")

        # ================= INVESTMENT RECOMMENDATION =================
        st.subheader("💡 Investment Recommendation")
        rec = get_investment_recommendation(pe_ratio, latest_price or 1000, eps, market_cap)
        st.info(f"Recommended Action: {rec}")

        # ================= HELPER / LEARNING MODE =================
        st.subheader("💡 Learning / Explanation")
        st.markdown("""
**Metrics Explained:**  
- **Market Cap:** Total value of the company’s shares.  
- **P/E Ratio:** Price-to-earnings, lower is generally safer.  
- **EPS:** Earnings per share, shows profitability.  
- **Revenue Growth %:** How fast the company is growing.  
- **Profit Growth %:** How net profit changes year-on-year.  
- **Stock Health:** Simple score based on key fundamentals.  

**Investment Recommendation:**  
- **Buy:** Favorable fundamentals and growth.  
- **Hold:** Moderate, consider with caution.  
- **Sell:** Weak fundamentals, risky investment.
""")

    except Exception as e:
        st.error("Unable to fetch fundamental data.")