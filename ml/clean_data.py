
import pandas as pd
import os
print("SCRIPT STARTED")

RAW_DATA_PATH = r"C:\stockmarket\data\raw data"
CLEAN_DATA_PATH = r"C:\stockmarket\data\clean"

print("RAW PATH EXISTS:", os.path.exists(RAW_DATA_PATH))
if os.path.exists(RAW_DATA_PATH):
    print("FILES INSIDE RAW:", os.listdir(RAW_DATA_PATH))


RAW_DATA_PATH = r"C:\stockmarket\data\raw data"
CLEAN_DATA_PATH = r"C:\stockmarket\data_clean"

print("RAW DATA FILES:", os.listdir(RAW_DATA_PATH))

os.makedirs(CLEAN_DATA_PATH, exist_ok=True)

def clean_stock_csv(file_path):
    df = pd.read_csv(file_path, header=[0, 1], index_col=0)

    # Flatten multi-level columns
    df.columns = [col[0].lower() for col in df.columns]

    # Reset index and force rename first column to date
    df.reset_index(inplace=True)
    df.rename(columns={df.columns[0]: 'date'}, inplace=True)

    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    df = df[required_cols]

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(inplace=True)

    return df

if __name__ == "__main__":
    for file in os.listdir(RAW_DATA_PATH):
        if file.endswith(".csv"):
            raw_path = os.path.join(RAW_DATA_PATH, file)
            print("Reading:", raw_path)

            clean_df = clean_stock_csv(raw_path)

            clean_path = os.path.join(CLEAN_DATA_PATH, file)
            clean_df.to_csv(clean_path, index=False)

            print("Saved:", clean_path)

