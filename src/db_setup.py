import sqlite3
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "db")
os.makedirs(DB_PATH, exist_ok=True)
DB_FILE = os.path.join(DB_PATH, "stock.db")

# Connect and create tables
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS portfolio (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    stock TEXT,
    quantity INTEGER,
    buy_price REAL,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
""")

conn.commit()
conn.close()
print(f"Database ready at {DB_FILE}")