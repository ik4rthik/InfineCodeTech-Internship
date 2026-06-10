"""
prediction/predictor.py
───────────────────────
Self-contained prediction layer for BenefiAI.

This module owns:
  • load_artifacts()         — load model + encoders + meta from disk (cached)
  • build_input_vector()     — encode raw user inputs into a feature DataFrame
  • run_prediction()         — call predict_proba(), return a rich result dict
  • save_prediction_to_db()  — persist every inference to the `predictions` table

Separation of concerns
───────────────────────
ml/model.py     → training, evaluation, model persistence (sklearn-level)
prediction/predictor.py → inference pipeline that Streamlit pages call

All encoding mirrors exactly what was done at training time so the model
receives the same feature representation it learned from.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.model        import load_model, model_exists, MODEL_PATH, ENCODERS_PATH, META_PATH
from ml.preprocessor import FEATURE_COLS, CATEGORICAL_COLS
from database.db_setup import get_connection

# ── Allowed option lists (must match training data exactly) ────────────────────
EMPLOYMENT_OPTIONS = ["Employed", "Unemployed", "Self-Employed", "Student", "Retired"]
EDUCATION_OPTIONS  = [
    "No Formal", "Primary", "Secondary",
    "Higher Secondary", "Graduate", "Post-Graduate",
]


# ══════════════════════════════════════════════════════════════════════════════
#  1. Load artefacts
# ══════════════════════════════════════════════════════════════════════════════

def load_artifacts() -> tuple[RandomForestClassifier, dict[str, LabelEncoder], dict]:
    """
    Load and return the trained model, encoders, and metadata.

    Returns
    -------
    model    : fitted RandomForestClassifier
    encoders : {column_name: LabelEncoder}  — must be applied to inputs
    meta     : dict from model_meta.json  (accuracy, trained_at, etc.)

    Raises
    ------
    FileNotFoundError if no model has been trained yet.
    """
    if not model_exists():
        raise FileNotFoundError(
            "No trained model found at ml/trained_models/rf_model.pkl. "
            "Please train the model first via the ML Training page."
        )
    return load_model()


# ══════════════════════════════════════════════════════════════════════════════
#  2. Build input feature vector
# ══════════════════════════════════════════════════════════════════════════════

def build_input_vector(
    raw_inputs: dict[str, Any],
    encoders:   dict[str, LabelEncoder],
) -> pd.DataFrame:
    """
    Convert raw user-supplied inputs into a one-row feature DataFrame
    that matches the exact schema the model was trained on.

    Steps
    ─────
    1. Copy the raw dict to avoid mutating caller state.
    2. Validate categorical values; fall back to the first known class if
       an unseen value is encountered (graceful degradation).
    3. Apply the saved LabelEncoder to each categorical column — this
       replicates the exact integer mapping used during training.
    4. Return a 1-row DataFrame with columns in FEATURE_COLS order.

    Parameters
    ----------
    raw_inputs : {
        "age":               int,
        "family_income":     float,
        "family_members":    int,
        "employment_status": str,   (e.g. "Unemployed")
        "education_level":   str,   (e.g. "Secondary")
        "disability_status": int,   (0 or 1)
    }
    encoders : fitted LabelEncoders from training artefacts

    Returns
    -------
    pd.DataFrame with shape (1, len(FEATURE_COLS))
    """
    row = dict(raw_inputs)  # defensive copy

    # Encode every categorical column
    for col in CATEGORICAL_COLS:
        if col not in row:
            continue
        le    = encoders[col]
        value = str(row[col]).strip()

        # Graceful unknown-category handling
        known_classes = set(le.classes_)
        if value not in known_classes:
            value = le.classes_[0]  # fallback to first class

        row[col] = int(le.transform([value])[0])

    # Ensure numeric types are correct
    row["age"]            = int(row["age"])
    row["family_income"]  = float(row["family_income"])
    row["family_members"] = int(row["family_members"])
    row["disability_status"] = int(row["disability_status"])

    # Build 1-row DataFrame in exact training column order
    X = pd.DataFrame([row])[FEATURE_COLS]
    return X


# ══════════════════════════════════════════════════════════════════════════════
#  3. Run prediction
# ══════════════════════════════════════════════════════════════════════════════

def run_prediction(
    raw_inputs: dict[str, Any],
    model:      RandomForestClassifier,
    encoders:   dict[str, LabelEncoder],
    meta:       dict,
) -> dict[str, Any]:
    """
    End-to-end inference for a single applicant.

    Uses predict_proba() to get calibrated probability scores,
    then derives the hard label (0/1) and confidence from those scores.

    How predict_proba() works
    ─────────────────────────
    Each of the 200 trees votes for a class. predict_proba() returns the
    fraction of trees that voted for each class:
      proba[0] = P(Not Eligible) = votes_for_0 / n_trees
      proba[1] = P(Eligible)     = votes_for_1 / n_trees
    The hard prediction is the class with the higher probability.
    Confidence = the probability of the winning class.

    Parameters
    ----------
    raw_inputs : user-supplied inputs (pre-encoding)
    model      : loaded RandomForestClassifier
    encoders   : loaded LabelEncoders
    meta       : model metadata (used to attach model_version to result)

    Returns
    -------
    {
        "label":             0 | 1,
        "verdict":           "Eligible" | "Not Eligible",
        "prob_eligible":     float  (0.0–1.0),
        "prob_not_eligible": float  (0.0–1.0),
        "confidence":        float  (probability of the winning class),
        "certainty_band":    "High" | "Moderate" | "Low",
        "model_version":     str,
        "feature_importance": dict,
        "predicted_at":      str  (ISO timestamp),
    }
    """
    # Build encoded feature vector
    X = build_input_vector(raw_inputs, encoders)

    # ── predict_proba() call ───────────────────────────────────────────────────
    # Returns array of shape (1, 2): [[p_not_eligible, p_eligible]]
    proba = model.predict_proba(X)[0]
    p_not_eligible = float(proba[0])
    p_eligible     = float(proba[1])

    # Hard label: argmax of probabilities
    label      = int(model.predict(X)[0])
    confidence = p_eligible if label == 1 else p_not_eligible

    # Certainty band — communicates how decisive the model's decision was
    if confidence >= 0.75:
        certainty_band = "High"
    elif confidence >= 0.55:
        certainty_band = "Moderate"
    else:
        certainty_band = "Low"

    return {
        "label":              label,
        "verdict":            "Eligible" if label == 1 else "Not Eligible",
        "prob_eligible":      round(p_eligible,     4),
        "prob_not_eligible":  round(p_not_eligible, 4),
        "confidence":         round(confidence,     4),
        "certainty_band":     certainty_band,
        "model_version":      meta.get("trained_at", "unknown"),
        "feature_importance": meta.get("feature_importance", {}),
        "predicted_at":       datetime.now().isoformat(timespec="seconds"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  4. Persist prediction to database
# ══════════════════════════════════════════════════════════════════════════════

def save_prediction_to_db(
    raw_inputs: dict[str, Any],
    result:     dict[str, Any],
    applicant_name: str = "Anonymous",
) -> tuple[bool, str]:
    """
    Insert a completed prediction run into the `predictions` table.

    This creates an audit trail of every inference made through the app,
    enabling retrospective review and model monitoring over time.

    Parameters
    ----------
    raw_inputs     : the original (pre-encoded) feature dict
    result         : dict returned by run_prediction()
    applicant_name : free-text name entered by the user (optional)

    Returns
    -------
    (success: bool, message: str)
    """
    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO predictions (
                    applicant_name,
                    age,
                    family_income,
                    family_members,
                    employment_status,
                    education_level,
                    disability_status,
                    predicted_label,
                    confidence_score,
                    model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                applicant_name,
                raw_inputs.get("age"),
                raw_inputs.get("family_income"),
                raw_inputs.get("family_members"),
                raw_inputs.get("employment_status"),
                raw_inputs.get("education_level"),
                raw_inputs.get("disability_status"),
                result["label"],
                result["confidence"],
                result["model_version"],
            ))
            conn.commit()
        return True, "Prediction saved to database."
    except Exception as exc:
        return False, f"Could not save prediction: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
#  5. Fetch prediction history
# ══════════════════════════════════════════════════════════════════════════════

def get_prediction_history(limit: int = 50) -> pd.DataFrame:
    """
    Return the most recent `limit` predictions from the predictions table.
    Used by the history tab on the Predict page.
    """
    with get_connection() as conn:
        df = pd.read_sql_query(
            f"""
            SELECT prediction_id, applicant_name, age, family_income,
                   family_members, employment_status, education_level,
                   disability_status, predicted_label, confidence_score,
                   model_version, predicted_at
            FROM predictions
            ORDER BY prediction_id DESC
            LIMIT {int(limit)}
            """,
            conn,
        )
    return df
