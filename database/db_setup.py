"""
database/db_setup.py
────────────────────
Initialises the SQLite database for BenefiAI.

Responsibilities
  • Create the `beneficiaries` table (schema mirrors the dataset columns).
  • Create the `predictions` table to log ML inference results.
  • Provide helper functions used across the app to open connections.

Run this file directly once to bootstrap the database:
    python database/db_setup.py
"""

import sqlite3
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent   # project root
DB_PATH  = BASE_DIR / "database" / "benefiai.db"


# ── Connection helper ──────────────────────────────────────────────────────────
def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with row-factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row          # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL") # better concurrency for Streamlit
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── Schema ─────────────────────────────────────────────────────────────────────
CREATE_BENEFICIARIES = """
CREATE TABLE IF NOT EXISTS beneficiaries (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    applicant_name      TEXT    NOT NULL,
    age                 INTEGER NOT NULL CHECK(age BETWEEN 1 AND 120),
    family_income       REAL    NOT NULL CHECK(family_income >= 0),
    family_members      INTEGER NOT NULL CHECK(family_members >= 1),
    employment_status   TEXT    NOT NULL,   -- 'Employed' | 'Unemployed' | 'Self-Employed' | 'Student' | 'Retired'
    education_level     TEXT    NOT NULL,   -- 'No Formal' | 'Primary' | 'Secondary' | 'Higher Secondary' | 'Graduate' | 'Post-Graduate'
    disability_status   INTEGER NOT NULL DEFAULT 0 CHECK(disability_status IN (0, 1)),
    eligibility_status  INTEGER NOT NULL DEFAULT 0 CHECK(eligibility_status IN (0, 1)),
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    applicant_name      TEXT    NOT NULL,
    age                 INTEGER,
    family_income       REAL,
    family_members      INTEGER,
    employment_status   TEXT,
    education_level     TEXT,
    disability_status   INTEGER,
    predicted_label     INTEGER,            -- 0 = Not Eligible, 1 = Eligible
    confidence_score    REAL,               -- probability from the ML model (0–1)
    model_version       TEXT,
    predicted_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_ben_eligibility ON beneficiaries(eligibility_status);",
    "CREATE INDEX IF NOT EXISTS idx_ben_employment  ON beneficiaries(employment_status);",
    "CREATE INDEX IF NOT EXISTS idx_pred_label      ON predictions(predicted_label);",
]


# ── Bootstrap ──────────────────────────────────────────────────────────────────
def initialise_db() -> None:
    """Create all tables and indexes. Safe to call multiple times (idempotent)."""
    print(f"[db_setup] Using database at: {DB_PATH}")
    with get_connection() as conn:
        conn.execute(CREATE_BENEFICIARIES)
        conn.execute(CREATE_PREDICTIONS)
        for idx_sql in CREATE_INDEXES:
            conn.execute(idx_sql)
        conn.commit()
    print("[db_setup] OK  Database initialised successfully.")


if __name__ == "__main__":
    initialise_db()
