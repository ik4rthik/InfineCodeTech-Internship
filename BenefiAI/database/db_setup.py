"""
database/db_setup.py
────────────────────
Initializes the SQLite database for BenefiAI.

Responsibilities:
  • Create the `beneficiaries` table if it does not already exist.
  • Provide helper functions for connecting to / closing the database.
  • Optionally seed the database from a CSV file.

The database file lives at: database/benefiai.db
"""

import sqlite3
import os
import pandas as pd

# ── Path resolution ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "benefiai.db")

# ── DDL ──────────────────────────────────────────────────────────────────────
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS beneficiaries (
    id                INTEGER  PRIMARY KEY AUTOINCREMENT,
    applicant_name    TEXT     NOT NULL,
    age               INTEGER  NOT NULL,
    family_income     REAL     NOT NULL,
    family_members    INTEGER  NOT NULL,
    employment_status TEXT     NOT NULL,
    education_level   TEXT     NOT NULL,
    disability_status TEXT     NOT NULL,
    eligibility_status TEXT    NOT NULL
);
"""


# ── Connection helpers ────────────────────────────────────────────────────────
def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with row_factory set to Row."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def close_connection(conn: sqlite3.Connection) -> None:
    """Safely close a database connection."""
    if conn:
        conn.close()


# ── Setup ─────────────────────────────────────────────────────────────────────
def initialize_database() -> None:
    """
    Create the database and beneficiaries table.
    Safe to call multiple times (uses IF NOT EXISTS).
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.executescript(CREATE_TABLE_SQL)
        conn.commit()
        print(f"[db_setup] Database ready at: {DB_PATH}")
    finally:
        close_connection(conn)


# ── Seed helper ───────────────────────────────────────────────────────────────
def seed_from_csv(csv_path: str) -> int:
    """
    Load records from a CSV file into the beneficiaries table.

    Parameters
    ----------
    csv_path : str
        Absolute or relative path to the CSV file produced by
        data/generate_dataset.py.

    Returns
    -------
    int
        Number of rows inserted.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Drop the 'id' column so SQLite auto-increments its own primary key.
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    conn = get_connection()
    try:
        df.to_sql(
            name="beneficiaries",
            con=conn,
            if_exists="append",   # append – preserves existing rows
            index=False,
        )
        conn.commit()
        inserted = len(df)
        print(f"[db_setup] Seeded {inserted} records into beneficiaries.")
        return inserted
    finally:
        close_connection(conn)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    initialize_database()

    # Optionally seed from the generated CSV
    default_csv = os.path.join(BASE_DIR, "..", "data", "beneficiaries.csv")
    if os.path.exists(default_csv):
        seed_from_csv(os.path.abspath(default_csv))
    else:
        print("[db_setup] No CSV found – run data/generate_dataset.py first.")
