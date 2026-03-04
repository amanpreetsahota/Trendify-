import pandas as pd
import os

CLEAN_DATA_PATH = r"C:\stockmarket\data_clean"
FEATURE_DATA_PATH = r"C:\stockmarket\data_features"

print(" CLEAN PATH EXISTS:", os.path.exists(CLEAN_DATA_PATH))

os.makedirs(FEATURE_DATA_PATH, exist_ok=True)

print("FILES IN CLEAN DATA:", os.listdir(CLEAN_DATA_PATH))

def add_features(file_path):
    df = pd.read_csv(file_path)

    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df.sort_values('date', inplace=True)

    df['daily_return'] = df['close'].pct_change()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    df['target_price'] = df['close'].shift(-1)
    df['target_trend'] = (df['target_price'] > df['close']).astype(int)

    df.dropna(inplace=True)
    return df


print("SCRIPT STARTED")

for file in os.listdir(CLEAN_DATA_PATH):
    if file.endswith(".csv"):
        clean_path = os.path.join(CLEAN_DATA_PATH, file)
        print("Processing:", clean_path)

        feature_df = add_features(clean_path)

        save_path = os.path.join(FEATURE_DATA_PATH, file)
        feature_df.to_csv(save_path, index=False)

        print("Saved:", save_path)

print(" FEATURE ENGINEERING COMPLETE")

