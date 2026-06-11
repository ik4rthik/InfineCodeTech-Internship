"""
database/crud.py
────────────────
Full CRUD interface for the `beneficiaries` SQLite table.

Functions
─────────
  fetch_all()                     -> pd.DataFrame
  fetch_by_id(record_id)          -> dict | None
  insert(data: dict)              -> int   (new row id)
  update(record_id, data: dict)   -> bool
  delete(record_id)               -> bool
  get_summary_stats()             -> dict
"""

import sqlite3
import pandas as pd
from database.db_setup import get_connection, close_connection, initialize_database

# Ensure table exists on first import
initialize_database()


# ── READ ──────────────────────────────────────────────────────────────────────

def fetch_all() -> pd.DataFrame:
    """Return every row from beneficiaries as a DataFrame."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM beneficiaries ORDER BY id DESC", conn
        )
        return df
    finally:
        close_connection(conn)


def fetch_by_id(record_id: int) -> dict | None:
    """Return a single record as a dict, or None if not found."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM beneficiaries WHERE id = ?", (record_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        close_connection(conn)


def get_summary_stats() -> dict:
    """
    Compute headline KPIs directly from the DB.

    Returns
    -------
    dict with keys:
        total, eligible, not_eligible, avg_income, min_income, max_income
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*)                                        AS total,
                SUM(CASE WHEN eligibility_status='Eligible'
                         THEN 1 ELSE 0 END)                    AS eligible,
                SUM(CASE WHEN eligibility_status='Not Eligible'
                         THEN 1 ELSE 0 END)                    AS not_eligible,
                ROUND(AVG(family_income), 2)                   AS avg_income,
                ROUND(MIN(family_income), 2)                   AS min_income,
                ROUND(MAX(family_income), 2)                   AS max_income
            FROM beneficiaries
        """)
        row = cursor.fetchone()
        return dict(row) if row else {}
    finally:
        close_connection(conn)


# ── CREATE ────────────────────────────────────────────────────────────────────

def insert(data: dict) -> int:
    """
    Insert a new beneficiary record.

    Parameters
    ----------
    data : dict
        Keys: applicant_name, age, family_income, family_members,
              employment_status, education_level, disability_status,
              eligibility_status

    Returns
    -------
    int – lastrowid of the newly inserted row
    """
    sql = """
        INSERT INTO beneficiaries
            (applicant_name, age, family_income, family_members,
             employment_status, education_level, disability_status,
             eligibility_status)
        VALUES
            (:applicant_name, :age, :family_income, :family_members,
             :employment_status, :education_level, :disability_status,
             :eligibility_status)
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(sql, data)
        conn.commit()
        return cursor.lastrowid
    finally:
        close_connection(conn)


# ── UPDATE ────────────────────────────────────────────────────────────────────

def update(record_id: int, data: dict) -> bool:
    """
    Update an existing beneficiary record by primary key.

    Parameters
    ----------
    record_id : int
    data      : dict  – same keys as insert()

    Returns
    -------
    bool – True if a row was modified
    """
    sql = """
        UPDATE beneficiaries SET
            applicant_name    = :applicant_name,
            age               = :age,
            family_income     = :family_income,
            family_members    = :family_members,
            employment_status = :employment_status,
            education_level   = :education_level,
            disability_status = :disability_status,
            eligibility_status = :eligibility_status
        WHERE id = :id
    """
    data_with_id = {**data, "id": record_id}
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(sql, data_with_id)
        conn.commit()
        return cursor.rowcount > 0
    finally:
        close_connection(conn)


# ── DELETE ────────────────────────────────────────────────────────────────────

def delete(record_id: int) -> bool:
    """
    Delete a beneficiary record by primary key.

    Returns
    -------
    bool – True if a row was removed
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM beneficiaries WHERE id = ?", (record_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        close_connection(conn)
