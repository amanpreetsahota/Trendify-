import sqlite3
import os

# =========================
# Database File
# =========================
DB_FILE = os.path.join("data", "trendify.db")

# =========================
# Initialize DB & Tables
# =========================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            password TEXT
        )
    """)
    c.execute("""
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

init_db()

# =========================
# User Functions
# =========================
def get_users():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name, password FROM users")
    rows = c.fetchall()
    conn.close()
    return {name: (user_id, password) for user_id, name, password in rows}

def add_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO users (name, password) VALUES (?, ?)", (username, password))
    conn.commit()
    conn.close()

# =========================
# Portfolio Functions
# =========================
def get_portfolio(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT stock, quantity, buy_price FROM portfolio WHERE user_id=?", (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def add_portfolio_entry(user_id, stock, quantity, buy_price):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # check if exists
    c.execute("SELECT quantity, buy_price FROM portfolio WHERE user_id=? AND stock=?", (user_id, stock))
    row = c.fetchone()
    if row:
        # update average buy price
        existing_qty, existing_price = row
        new_qty = existing_qty + quantity
        new_avg_price = (existing_price * existing_qty + buy_price * quantity) / new_qty
        c.execute("UPDATE portfolio SET quantity=?, buy_price=? WHERE user_id=? AND stock=?",
                  (new_qty, new_avg_price, user_id, stock))
    else:
        c.execute("INSERT INTO portfolio (user_id, stock, quantity, buy_price) VALUES (?, ?, ?, ?)",
                  (user_id, stock, quantity, buy_price))
    conn.commit()
    conn.close()

def update_portfolio_entry(user_id, stock, quantity, buy_price):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE portfolio SET quantity=?, buy_price=? WHERE user_id=? AND stock=?",
              (quantity, buy_price, user_id, stock))
    conn.commit()
    conn.close()

def delete_portfolio_entry(user_id, stock):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM portfolio WHERE user_id=? AND stock=?", (user_id, stock))
    conn.commit()
    conn.close()