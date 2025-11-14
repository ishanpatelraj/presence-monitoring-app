# db_utils.py
import sqlite3
from pathlib import Path
import numpy as np

DB_FILE = Path("users.db")

def init_db():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            status TEXT,
            info TEXT,
            ts TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_user(name: str, embedding: np.ndarray):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    emb_blob = embedding.astype(np.float32).tobytes()
    cur.execute("INSERT INTO users (name, embedding) VALUES (?, ?)", (name, emb_blob))
    conn.commit()
    conn.close()

def get_all_users():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT id, name, embedding FROM users")
    rows = cur.fetchall()
    conn.close()
    users = []
    for uid, name, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        users.append({"id": uid, "name": name, "embedding": emb})
    return users

def log_event(user: str, status: str, info: str = ""):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT INTO logs (user, status, info) VALUES (?, ?, ?)", (user, status, info))
    conn.commit()
    conn.close()

def get_logs(limit: int = 100):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT id, user, status, info, ts FROM logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows
