"""
modules/prediction.py
═════════════════════
BenefiAI – Prediction Bridge

Connects the Streamlit UI to the ML model.
Handles model loading, single-applicant prediction, batch prediction,
and logging every prediction to an SQLite audit table.

Functions
─────────
  run_prediction(applicant_dict)    → result dict
  batch_predict(df)                 → DataFrame with added columns
  log_prediction(applicant, result) → None   (writes to SQLite)
  get_prediction_log()              → pd.DataFrame
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime

from modules.ml_model import (
    load_model, predict_single, model_exists, MODEL_PATH,
)
from database.db_setup import get_connection, close_connection, initialize_database

# ── Ensure predictions log table exists ──────────────────────────────────────
_CREATE_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS prediction_log (
    id              INTEGER  PRIMARY KEY AUTOINCREMENT,
    applicant_name  TEXT,
    age             INTEGER,
    family_income   REAL,
    family_members  INTEGER,
    employment_status TEXT,
    education_level   TEXT,
    disability_status TEXT,
    predicted_label TEXT,
    confidence      REAL,
    flag_count      INTEGER,
    predicted_at    TEXT
);
"""

def _ensure_log_table() -> None:
    initialize_database()   # ensure beneficiaries table exists too
    conn = get_connection()
    try:
        conn.executescript(_CREATE_LOG_TABLE)
        conn.commit()
    finally:
        close_connection(conn)

_ensure_log_table()


# ── Module-level model cache (avoid reloading on every call) ──────────────────
_BUNDLE_CACHE: dict | None = None

def _get_bundle() -> dict | None:
    """Return cached model bundle, loading from disk if needed."""
    global _BUNDLE_CACHE
    if _BUNDLE_CACHE is None:
        _BUNDLE_CACHE = load_model()
    return _BUNDLE_CACHE

def refresh_bundle() -> None:
    """Force-reload the model from disk (call after retraining)."""
    global _BUNDLE_CACHE
    _BUNDLE_CACHE = load_model()


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def run_prediction(applicant_dict: dict) -> dict:
    """
    Predict eligibility for one applicant and log the result.

    Parameters
    ──────────
    applicant_dict : dict
        Required keys:
          applicant_name, age, family_income, family_members,
          employment_status, education_level, disability_status

    Returns
    ───────
    {
      "label":      "Eligible" | "Not Eligible",
      "confidence": float,
      "flags":      dict of vulnerability indicators,
      "per_capita_income": float,
      "timestamp":  ISO-8601 string,
      "error":      str   ← only if model not loaded
    }
    """
    bundle = _get_bundle()
    if bundle is None:
        return {
            "label":      "Unknown",
            "confidence": 0.0,
            "error":      "Model not trained yet. Go to ML Insights and click Train.",
        }

    result = predict_single(bundle, applicant_dict)
    result["timestamp"] = datetime.now().isoformat(timespec="seconds")

    # Persist to audit log
    log_prediction(applicant_dict, result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run predictions on every row in df.

    Adds two columns:
      predicted_label – "Eligible" | "Not Eligible"
      confidence      – probability of being Eligible (0.0–1.0)

    Rows that fail (e.g. unknown category value) are marked as "Error".
    """
    bundle = _get_bundle()
    if bundle is None:
        df = df.copy()
        df["predicted_label"] = "Model Not Trained"
        df["confidence"]      = 0.0
        return df

    labels      = []
    confidences = []

    for _, row in df.iterrows():
        try:
            res = predict_single(bundle, row.to_dict())
            labels.append(res["label"])
            confidences.append(res["confidence"])
        except Exception:
            labels.append("Error")
            confidences.append(0.0)

    out = df.copy()
    out["predicted_label"] = labels
    out["confidence"]      = confidences
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════

def log_prediction(applicant: dict, result: dict) -> None:
    """
    Write one prediction to the prediction_log SQLite table.

    Every call from run_prediction() triggers this automatically,
    providing a complete audit trail of all predictions made.
    """
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO prediction_log
                (applicant_name, age, family_income, family_members,
                 employment_status, education_level, disability_status,
                 predicted_label, confidence, flag_count, predicted_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            applicant.get("applicant_name", ""),
            applicant.get("age"),
            applicant.get("family_income"),
            applicant.get("family_members"),
            applicant.get("employment_status", ""),
            applicant.get("education_level", ""),
            applicant.get("disability_status", ""),
            result.get("label", ""),
            result.get("confidence", 0.0),
            result.get("flags", {}).get("total_flags", 0),
            result.get("timestamp", datetime.now().isoformat()),
        ))
        conn.commit()
    finally:
        close_connection(conn)


def get_prediction_log() -> pd.DataFrame:
    """Return the full prediction audit log as a DataFrame."""
    conn = get_connection()
    try:
        return pd.read_sql_query(
            "SELECT * FROM prediction_log ORDER BY id DESC", conn
        )
    finally:
        close_connection(conn)
