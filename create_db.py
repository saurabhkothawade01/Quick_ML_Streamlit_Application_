import sqlite3
from hashlib import sha256

conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()

def create_users_table():
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()

def create_datasets_table():
    c.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            dataset_name TEXT NOT NULL,
            dataset_file TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()

def create_models_table():
    c.execute('''
        CREATE TABLE IF NOT EXISTS models (
        model_id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER,
        model_name TEXT,
        ml_model BLOB,
        FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
        )
    ''')
    conn.commit()
    
def hash_password(password):
    return sha256(password.encode()).hexdigest()

def register_user(username, password):
    try:
        hashed_password = hash_password(password)
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                  (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    hashed_password = hash_password(password)
    c.execute('SELECT id, username FROM users WHERE username = ? AND password = ?', 
              (username, hashed_password))
    return c.fetchone()

create_users_table()
create_datasets_table()
create_models_table()