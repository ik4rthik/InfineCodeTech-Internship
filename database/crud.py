"""
database/crud.py
────────────────
All Create / Read / Update / Delete operations for the `beneficiaries` table.
Every public function returns plain Python types or a pandas DataFrame so the
Streamlit layer stays free of SQL.
"""

from __future__ import annotations

import pandas as pd

from database.db_setup import get_connection


# ── READ ───────────────────────────────────────────────────────────────────────

def get_all_beneficiaries() -> pd.DataFrame:
    """Return every row in the beneficiaries table as a DataFrame."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM beneficiaries ORDER BY id DESC", conn
        )
    return df


def get_beneficiary_by_id(record_id: int) -> dict | None:
    """Return a single beneficiary row as a dict, or None if not found."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM beneficiaries WHERE id = ?", (record_id,)
        ).fetchone()
    return dict(row) if row else None


def get_summary_stats() -> dict:
    """
    Compute headline statistics used by the Dashboard cards.

    Returns
    -------
    dict with keys:
        total, eligible, ineligible, avg_income,
        eligible_pct, ineligible_pct
    """
    with get_connection() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                                    AS total,
                SUM(eligibility_status)                    AS eligible,
                COUNT(*) - SUM(eligibility_status)         AS ineligible,
                ROUND(AVG(family_income), 2)               AS avg_income
            FROM beneficiaries
        """).fetchone()

    total     = row["total"]     or 0
    eligible  = row["eligible"]  or 0
    ineligible= row["ineligible"]or 0
    avg_income= row["avg_income"]or 0.0

    return {
        "total":          total,
        "eligible":       eligible,
        "ineligible":     ineligible,
        "avg_income":     avg_income,
        "eligible_pct":   round(eligible   / total * 100, 1) if total else 0,
        "ineligible_pct": round(ineligible / total * 100, 1) if total else 0,
    }


# ── CREATE ─────────────────────────────────────────────────────────────────────

def add_beneficiary(
    applicant_name: str,
    age: int,
    family_income: float,
    family_members: int,
    employment_status: str,
    education_level: str,
    disability_status: int,
    eligibility_status: int,
) -> tuple[bool, str]:
    """
    Insert a new beneficiary record.

    Returns
    -------
    (success: bool, message: str)
    """
    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO beneficiaries
                    (applicant_name, age, family_income, family_members,
                     employment_status, education_level,
                     disability_status, eligibility_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                applicant_name, age, family_income, family_members,
                employment_status, education_level,
                disability_status, eligibility_status,
            ))
            conn.commit()
        return True, "Beneficiary added successfully."
    except Exception as exc:
        return False, f"Error adding beneficiary: {exc}"


# ── UPDATE ─────────────────────────────────────────────────────────────────────

def update_beneficiary(
    record_id: int,
    applicant_name: str,
    age: int,
    family_income: float,
    family_members: int,
    employment_status: str,
    education_level: str,
    disability_status: int,
    eligibility_status: int,
) -> tuple[bool, str]:
    """
    Update an existing beneficiary record by primary key.

    Returns
    -------
    (success: bool, message: str)
    """
    try:
        with get_connection() as conn:
            cursor = conn.execute("""
                UPDATE beneficiaries SET
                    applicant_name    = ?,
                    age               = ?,
                    family_income     = ?,
                    family_members    = ?,
                    employment_status = ?,
                    education_level   = ?,
                    disability_status = ?,
                    eligibility_status= ?
                WHERE id = ?
            """, (
                applicant_name, age, family_income, family_members,
                employment_status, education_level,
                disability_status, eligibility_status,
                record_id,
            ))
            conn.commit()

        if cursor.rowcount == 0:
            return False, f"No record found with ID {record_id}."
        return True, f"Record #{record_id} updated successfully."
    except Exception as exc:
        return False, f"Error updating record: {exc}"


# ── DELETE ─────────────────────────────────────────────────────────────────────

def delete_beneficiary(record_id: int) -> tuple[bool, str]:
    """
    Delete a beneficiary record by primary key.

    Returns
    -------
    (success: bool, message: str)
    """
    try:
        with get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM beneficiaries WHERE id = ?", (record_id,)
            )
            conn.commit()

        if cursor.rowcount == 0:
            return False, f"No record found with ID {record_id}."
        return True, f"Record #{record_id} deleted successfully."
    except Exception as exc:
        return False, f"Error deleting record: {exc}"
