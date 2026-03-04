import streamlit as st

def generate_recommendation(last_close, predicted_close, pe_ratio, revenue_growth, profit_growth):
    
    score = 0
    reasons = []

    # ML Prediction
    if predicted_close > last_close:
        score += 1
        reasons.append("Model predicts upward price movement.")
    else:
        reasons.append("Model predicts downward/flat movement.")

    # Valuation
    if pe_ratio is not None and pe_ratio < 25:
        score += 1
        reasons.append("P/E ratio indicates reasonable valuation.")
    else:
        reasons.append("Stock may be overvalued.")

    # Growth
    if revenue_growth is not None and revenue_growth > 10:
        score += 1
        reasons.append("Strong revenue growth.")
    
    if profit_growth is not None and profit_growth > 10:
        score += 1
        reasons.append("Strong profit growth.")

    # Final Decision
    if score >= 3:
        recommendation = "BUY 🟢"
    elif score == 2:
        recommendation = "HOLD 🟡"
    else:
        recommendation = "SELL 🔴"

    return recommendation, reasons