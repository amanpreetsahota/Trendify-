import pandas as pd
import numpy as np
import streamlit as st

def show_price_prediction(df, reg_model, days=7):
    st.subheader("🔮 Next 7 Days Price Prediction")

    future_prices = []
    temp_df = df.copy()

    for i in range(days):
        latest = temp_df.iloc[-1]

        features = [
            "open", "high", "low", "close", "volume",
            "daily_return", "sma_10", "sma_50"
        ]

        X = latest[features].values.reshape(1, -1)
        pred_price = reg_model.predict(X)[0]

        # Add small volatility for realism (optional but recommended)
        volatility = np.random.normal(0, 0.003)
        pred_price = pred_price * (1 + volatility)

        future_prices.append(pred_price)

        # Create new synthetic row
        new_row = latest.copy()

        new_row["open"] = latest["close"]
        new_row["high"] = pred_price * 1.01
        new_row["low"] = pred_price * 0.99
        new_row["close"] = pred_price
        new_row["volume"] = latest["volume"]

        new_row["daily_return"] = (
            pred_price - latest["close"]
        ) / latest["close"]

        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)

        # 🔥 Recalculate SMAs so they evolve
        temp_df["sma_10"] = temp_df["close"].rolling(10).mean()
        temp_df["sma_50"] = temp_df["close"].rolling(50).mean()

    future_df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(days)],
        "Predicted Close": [round(price, 2) for price in future_prices]
    })

    st.dataframe(future_df)
    st.area_chart(future_df.set_index("Day"))