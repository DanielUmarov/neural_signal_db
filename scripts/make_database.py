# make_database.py
import sqlite3

conn = sqlite3.connect("neural_signal_db.db")
cur = conn.cursor()

# Create main recordings table
cur.execute("""
CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT,
    paradigm TEXT,
    run TEXT,
    file_path TEXT,
    sampling_rate REAL,
    n_channels INTEGER,
    duration REAL,
    notes TEXT
);
""")

# Optional: create a table for metadata sidecars
cur.execute("""
CREATE TABLE IF NOT EXISTS metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id INTEGER,
    key TEXT,
    value TEXT,
    FOREIGN KEY (recording_id) REFERENCES recordings(id)
);
""")

conn.commit()
conn.close()
print("Database initialized: neural_signal_db.db")
