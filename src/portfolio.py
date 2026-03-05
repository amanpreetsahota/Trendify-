import streamlit as st
import matplotlib.pyplot as plt
from db_manager import get_portfolio, add_portfolio_entry, update_portfolio_entry, delete_portfolio_entry
import yfinance as yf

# ==============================
# Cached Live Price
# ==============================
@st.cache_data(ttl=300)
def get_live_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        if not data.empty:
            return float(data["Close"].iloc[-1])
        return None
    except:
        return None

# ==============================
# Tracked Stocks
# ==============================
STOCK_LIST = {"Reliance Industries": "RELIANCE.NS",
            "TCS": "TCS.NS",
             "Infosys": "INFY.NS",
            "HDFC Bank": "HDFCBANK.NS",
             "ICICI Bank": "ICICIBANK.NS"
             }
# Portfolio UI
def show_portfolio():
    st.subheader("💼 Your Portfolio")

    user_id = st.session_state.user_id
    portfolio = get_portfolio(user_id)

    # ===== Add / Update Stock =====
    st.markdown("### ➕ Add / Update Stock")
    col1, col2, col3 = st.columns(3)
    with col1:
        stock_name = st.selectbox("Select Stock", list(STOCK_LIST.keys()))
        symbol = STOCK_LIST[stock_name]
    with col2:
        buy_price = st.number_input("Buy Price (₹)", min_value=0.0, format="%.2f")
    with col3:
        quantity = st.number_input("Quantity", min_value=1, step=1)

    if st.button("Add / Update"):
        if symbol in [s for s, _, _ in portfolio]:
            # Update existing
            for s, qty, price in portfolio:
                if s == symbol:
                    new_qty = qty + quantity
                    new_avg_price = (price * qty + buy_price * quantity) / new_qty
                    update_portfolio_entry(user_id, symbol, new_qty, new_avg_price)
                    st.success(f"Updated {stock_name} in portfolio!")
                    break
        else:
            # Add new
            add_portfolio_entry(user_id, symbol, quantity, buy_price)
            st.success(f"Added {stock_name} to portfolio!")
        st.rerun()

    st.markdown("---")

    # ===== Portfolio Table & Charts =====
    portfolio = get_portfolio(user_id)
    if portfolio:
        total_invested = 0
        total_current = 0
        allocation_data = {}
        pl_data = {}

        for symbol, qty, buy_price in portfolio:
            current_price = get_live_price(symbol) or buy_price
            invested = qty * buy_price
            current_value = qty * current_price
            pl = current_value - invested
            return_percent = (pl / invested * 100) if invested != 0 else 0

            total_invested += invested
            total_current += current_value
            allocation_data[symbol] = current_value
            pl_data[symbol] = pl

            col1, col2 = st.columns([3,1])
            with col1:
                st.markdown(f"**{symbol}**")
                st.write(f"Quantity: {qty}")
                st.write(f"Avg Buy Price: ₹ {buy_price:.2f}")
                st.write(f"Current Price: ₹ {current_price:.2f}")
                if pl >=0:
                    st.success(f"P/L: ₹ {pl:.2f} ({return_percent:.2f}%)")
                else:
                    st.error(f"P/L: ₹ {pl:.2f} ({return_percent:.2f}%)")
            with col2:
                if st.button("❌ Delete", key=symbol):
                    delete_portfolio_entry(user_id, symbol)
                    st.rerun()

            st.markdown("---")

        # ===== Portfolio Summary =====
        net_pl = total_current - total_invested
        net_percent = (net_pl / total_invested * 100) if total_invested != 0 else 0
        st.subheader("📊 Portfolio Summary")
        st.write(f"Total Invested: ₹ {total_invested:.2f}")
        st.write(f"Total Current Value: ₹ {total_current:.2f}")
        if net_pl >= 0:
            st.success(f"Net P/L: ₹ {net_pl:.2f} ({net_percent:.2f}%)")
        else:
            st.error(f"Net P/L: ₹ {net_pl:.2f} ({net_percent:.2f}%)")

        # ===== Allocation Pie Chart =====
        st.subheader("📈 Portfolio Allocation")
        fig1, ax1 = plt.subplots()
        ax1.pie(allocation_data.values(), labels=allocation_data.keys(), autopct="%1.1f%%")
        ax1.axis("equal")
        st.pyplot(fig1)

        # ===== Profit/Loss Bar Chart =====
        st.subheader("💹 Profit/Loss per Stock")
        fig2, ax2 = plt.subplots()
        colors = ['green' if val>=0 else 'red' for val in pl_data.values()]
        bars = ax2.bar(pl_data.keys(), pl_data.values(), color=colors)
        ax2.set_ylabel("Profit / Loss (₹)")
        ax2.set_xticks(range(len(pl_data)))
        ax2.set_xticklabels(pl_data.keys(), rotation=45)
        st.pyplot(fig2)
        for i, val in enumerate(pl_data.values()):
            ax2.text(i, val, f"{val:.0f}", ha='center', va='bottom' if val>=0 else 'top')

    else:
        st.info("Portfolio empty.")