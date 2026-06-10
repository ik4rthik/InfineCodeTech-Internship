"""
ml/preprocessor.py
──────────────────
Data preprocessing pipeline for BenefiAI ML training.

Why each step matters
──────────────────────
  1. LOAD       Pull raw rows from SQLite; keep only the columns the model needs.
  2. CLEAN      Strip whitespace from text, drop rows with impossible values
                (negative income, age out of range) so garbage doesn't bias training.
  3. DUPLICATES Remove exact duplicate rows; they inflate accuracy and skew metrics.
  4. MISSING    Fill gaps rather than drop rows: median for numeric columns
                (robust to outliers), mode for categoricals (most common class).
  5. ENCODE     sklearn needs numbers. LabelEncoder maps each category string to a
                stable integer. We keep the fitted encoders so the exact same
                mapping is applied at prediction time.
  6. SPLIT X/y  Separate features (inputs) from the target (eligibility_status).

Public API
──────────
  preprocess(df) -> (X, y, encoders, report)
  load_raw_data() -> pd.DataFrame
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Project root on the path ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db_setup import get_connection

# ── Column groups ──────────────────────────────────────────────────────────────
# These are the only columns used for training.
FEATURE_COLS = [
    "age",
    "family_income",
    "family_members",
    "employment_status",    # categorical → encoded
    "education_level",      # categorical → encoded
    "disability_status",    # already 0/1
]
TARGET_COL       = "eligibility_status"
CATEGORICAL_COLS = ["employment_status", "education_level"]
NUMERIC_COLS     = ["age", "family_income", "family_members"]

# Allowed value ranges for basic sanity checks
VALID_RANGES = {
    "age":            (1,   120),
    "family_income":  (0,   1e8),
    "family_members": (1,   50),
}


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 – Load raw data from SQLite
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_data() -> pd.DataFrame:
    """
    Pull every row from the `beneficiaries` table.

    Returns a DataFrame with all columns (including id, applicant_name,
    created_at) — the preprocessing step will drop non-feature columns.
    """
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM beneficiaries", conn)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 – Data cleaning
# ══════════════════════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean the raw DataFrame:
      • Strip leading/trailing whitespace from all string columns.
      • Drop rows whose numeric values fall outside the allowed range.
      • Standardise category strings (title-case, collapse known variants).

    Returns (cleaned_df, report) where report contains counts of changes made.
    """
    report: dict[str, Any] = {}
    original_len = len(df)
    df = df.copy()

    # ── Strip whitespace from every string column ──────────────────────────────
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # ── Standardise text category columns ─────────────────────────────────────
    # e.g. "self-employed" → "Self-Employed" so encoders see consistent keys
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].str.title()

    # ── Remove out-of-range rows ───────────────────────────────────────────────
    mask_valid = pd.Series([True] * len(df), index=df.index)
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            in_range = df[col].between(lo, hi)
            mask_valid &= in_range

    n_invalid = (~mask_valid).sum()
    df = df[mask_valid].reset_index(drop=True)
    report["rows_removed_invalid_range"] = int(n_invalid)
    report["rows_after_cleaning"]        = len(df)
    report["original_rows"]             = int(original_len)

    return df, report


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 – Remove duplicates
# ══════════════════════════════════════════════════════════════════════════════

def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Drop rows that are exact duplicates across all feature + target columns.

    We do NOT treat `id` or `applicant_name` as part of the duplicate key
    because two real applicants could coincidentally share all other attributes.
    We check only the meaningful columns.
    """
    check_cols = FEATURE_COLS + [TARGET_COL]
    before = len(df)
    df = df.drop_duplicates(subset=check_cols).reset_index(drop=True)
    removed = before - len(df)
    return df, {"duplicates_removed": removed, "rows_after_dedup": len(df)}


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 – Handle missing values
# ══════════════════════════════════════════════════════════════════════════════

def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing values without dropping rows (keeps the dataset large).

    Strategy:
      • Numeric  → median  (median is robust to extreme outliers unlike mean)
      • Categorical → mode  (most frequent category)
      • Target (eligibility_status) → rows with missing target are dropped
        because we cannot train on an unknown label.
    """
    report: dict[str, Any] = {}
    df = df.copy()

    # Drop rows where the target is missing — we can't learn from them
    target_missing = df[TARGET_COL].isna().sum()
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    report["target_rows_dropped"] = int(target_missing)

    # Impute numeric columns with column median
    for col in NUMERIC_COLS:
        if col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            report[f"{col}_imputed"] = n_missing

    # Impute categorical columns with column mode
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
            report[f"{col}_imputed"] = n_missing

    report["rows_after_imputation"] = len(df)
    return df, report


# ══════════════════════════════════════════════════════════════════════════════
#  Step 5 – Label-encode categorical features
# ══════════════════════════════════════════════════════════════════════════════

def encode_features(
    df: pd.DataFrame,
    existing_encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder], dict]:
    """
    Convert categorical string columns to integer codes using LabelEncoder.

    Why LabelEncoder (not OneHotEncoder)?
    • RandomForest can handle ordinal-style encoding.
    • OneHot would create many sparse columns for little gain here.

    Parameters
    ----------
    df               : cleaned DataFrame
    existing_encoders: if provided, use these fitted encoders (prediction path)
                       instead of fitting new ones (training path).

    Returns
    -------
    encoded_df : DataFrame with categorical columns replaced by integer codes
    encoders   : {col_name: fitted LabelEncoder}  — save these with the model!
    report     : human-readable encoding summary
    """
    df = df.copy()
    encoders: dict[str, LabelEncoder] = existing_encoders or {}
    report: dict[str, Any] = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue

        if existing_encoders and col in existing_encoders:
            # Prediction path: use the previously fitted encoder
            le = existing_encoders[col]
            # Handle unseen categories gracefully
            known = set(le.classes_)
            df[col] = df[col].apply(lambda v: v if v in known else le.classes_[0])
            df[col] = le.transform(df[col])
        else:
            # Training path: fit a new encoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        report[col] = {
            "classes": list(encoders[col].classes_),
            "mapping": {cls: int(code) for code, cls in
                        enumerate(encoders[col].classes_)},
        }

    return df, encoders, report


# ══════════════════════════════════════════════════════════════════════════════
#  Step 6 – Split features and target
# ══════════════════════════════════════════════════════════════════════════════

def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate the feature matrix X from the target vector y.

    Features used
    ─────────────
    age               : numeric – younger applicants may be students
    family_income     : numeric – primary eligibility driver
    family_members    : numeric – larger families more likely eligible
    employment_status : encoded – unemployed/student more likely eligible
    education_level   : encoded – lower education → higher need
    disability_status : binary  – direct eligibility boost
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  Master pipeline
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(
    df: pd.DataFrame | None = None,
    existing_encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder], dict]:
    """
    Run all preprocessing steps in sequence.

    Parameters
    ----------
    df               : raw DataFrame; if None, loads from SQLite.
    existing_encoders: pass saved encoders when calling from the predict path.

    Returns
    -------
    X        : feature matrix (numeric, encoded)
    y        : target vector (0/1)
    encoders : fitted LabelEncoders for every categorical column
    full_report : step-by-step counts for the UI to display
    """
    if df is None:
        df = load_raw_data()

    full_report: dict[str, Any] = {"initial_rows": len(df)}

    # Step 2 – Clean
    df, clean_rpt = clean_data(df)
    full_report["cleaning"] = clean_rpt

    # Step 3 – Deduplicate
    df, dedup_rpt = remove_duplicates(df)
    full_report["deduplication"] = dedup_rpt

    # Step 4 – Impute missing values
    df, miss_rpt = handle_missing_values(df)
    full_report["missing_values"] = miss_rpt

    # Step 5 – Encode
    df, encoders, enc_rpt = encode_features(df, existing_encoders)
    full_report["encoding"] = enc_rpt

    # Step 6 – Split X / y
    X, y = split_features_target(df)
    full_report["final_rows"]    = len(X)
    full_report["feature_cols"]  = FEATURE_COLS
    full_report["class_balance"] = {
        "eligible":     int((y == 1).sum()),
        "not_eligible": int((y == 0).sum()),
    }

    return X, y, encoders, full_report
