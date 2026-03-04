import yfinance as yf
import os
stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

if not os.path.exists("data"):
    os.makedirs("data")

for s in stocks:
    file_path = f"data/{s}.csv"
    try:
        if not os.path.exists(file_path): 
            print(f"Downloading: {s} ...")
            df = yf.download(s, period="5y", threads=False)
            df.to_csv(file_path)
            print("Saved:", file_path)
        else:
            print("Already exists, skipped:", s)
    except PermissionError:
        print(f"Permission denied, cannot save: {s}")
    except Exception as e:
        print(f"Failed to download {s}: {e}")



