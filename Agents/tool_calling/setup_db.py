"""Create a tiny SQLite shop database with two customers and three orders.

Run this once before the tool-calling examples:

    python setup_db.py
"""

import os
import sqlite3

# Start fresh each time so the examples are reproducible.
if os.path.exists("shop.db"):
    os.remove("shop.db")

conn = sqlite3.connect("shop.db")

conn.executescript("""
CREATE TABLE customers (
    id     INTEGER PRIMARY KEY,
    name   TEXT,
    email  TEXT UNIQUE
);
CREATE TABLE orders (
    id           INTEGER PRIMARY KEY,
    customer_id  INTEGER REFERENCES customers(id),
    product      TEXT,
    quantity     INTEGER,
    status       TEXT,
    order_date   TEXT
);
INSERT INTO customers (name, email) VALUES
    ('Jane Doe',     'jane@example.com'),
    ('Nikos Pappas', 'nikos@example.com');
INSERT INTO orders (customer_id, product, quantity, status, order_date) VALUES
    (1, 'Mechanical Keyboard', 1, 'shipped',    '2026-06-10'),
    (1, 'USB-C Cable',         3, 'delivered',  '2026-05-22'),
    (2, 'Monitor Stand',       1, 'processing', '2026-06-18');
""")
conn.commit()
conn.close()

print("Created shop.db with 2 customers and 3 orders.")
