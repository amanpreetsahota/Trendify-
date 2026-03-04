import pandas as pd

#file paths
INPUT_FILE = "data/HDFCBANK.NS.csv"
OUTPUT_FILE = "data/HDFCBANK_clean.csv"

# raw data
# Yahoo Finance CSV has multi-level headers
df = pd.read_csv(INPUT_FILE, header=[0, 1], index_col=0)

print("Before cleaning:")
print(df.head())
print(df.shape)

#FIX COLUMN NAMES
# Flatten multi-index columns
df.columns = [col[0] for col in df.columns]

# FIX DATE
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)


df.dropna(inplace=True) #handle missing values 

print("\nAfter cleaning:")
print(df.head())
print(df.shape)


df.to_csv(OUTPUT_FILE)  #save clean data

print("\n Clean dataset saved as:", OUTPUT_FILE)
