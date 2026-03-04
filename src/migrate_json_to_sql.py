import json
import sqlite3

# Absolute paths
USERS_FILE = r"C:\stockmarket\data\users.json"
PORTFOLIO_FILE = r"C:\stockmarket\data\portfolios.json"
DB_FILE = r"C:\stockmarket\db\stock.db"

conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()

# ----------------- CREATE TABLES -----------------
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    email TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS portfolio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    stock TEXT,
    quantity REAL,
    buy_price REAL,
    FOREIGN KEY (user_id) REFERENCES users(id)
)
""")

# ----------------- MIGRATE USERS -----------------
with open(USERS_FILE, "r") as f:
    users = json.load(f)

for idx, (username, password) in enumerate(users.items(), start=1):
    cur.execute(
        "INSERT OR IGNORE INTO users (id, name, email) VALUES (?, ?, ?)",
        (idx, username, password)
    )

# Map usernames to user_id
cur.execute("SELECT id, name FROM users")
user_map = {name: uid for uid, name in cur.fetchall()}

# ----------------- MIGRATE PORTFOLIO -----------------
with open(PORTFOLIO_FILE, "r") as f:
    portfolios = json.load(f)

for username, entries in portfolios.items():
    user_id = user_map.get(username)
    if not user_id:
        continue
    for e in entries:
        cur.execute(
            "INSERT INTO portfolio (user_id, stock, quantity, buy_price) VALUES (?, ?, ?, ?)",
            (user_id, e["stock"], e["quantity"], e["buy_price"])
        )

conn.commit()
conn.close()

print("✅ Migration complete. Users and portfolio are now in SQL!")