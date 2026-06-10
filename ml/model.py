"""
ml/model.py
───────────
RandomForest model — training, evaluation, persistence.

Why RandomForestClassifier?
────────────────────────────
• Handles mixed numeric + encoded categorical features without scaling.
• Provides native feature_importances_ for free explainability.
• Robust to the ~10% label noise injected during dataset generation.
• Resistant to overfitting via bagging (averaging N decorrelated trees).
• Works well out-of-the-box for small tabular datasets (300 rows).

Hyperparameters chosen
───────────────────────
  n_estimators  = 200   → 200 trees; more stable than the default 100
  max_depth     = None  → trees grow until pure leaves (forest handles overfitting)
  min_samples_split = 4 → avoid splits on fewer than 4 samples (prevents noise fit)
  class_weight  = "balanced" → compensates for the 64/36 class imbalance
  random_state  = 42    → reproducibility

Saved artefacts (under ml/trained_models/)
───────────────────────────────────────────
  rf_model.pkl    — the fitted RandomForestClassifier
  encoders.pkl    — {col: LabelEncoder}; must match the model's training schema
  model_meta.json — training metadata (date, accuracy, n_samples, …)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MODELS_DIR    = PROJECT_ROOT / "ml" / "trained_models"
MODEL_PATH    = MODELS_DIR / "rf_model.pkl"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"
META_PATH     = MODELS_DIR / "model_meta.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
RF_PARAMS = dict(
    n_estimators      = 200,
    max_depth         = None,
    min_samples_split = 4,
    min_samples_leaf  = 2,
    max_features      = "sqrt",   # sqrt(n_features) considered per split
    class_weight      = "balanced",
    random_state      = 42,
    n_jobs            = -1,       # use all CPU cores
)

TEST_SIZE   = 0.20   # 80 % train / 20 % test
RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
#  Train
# ══════════════════════════════════════════════════════════════════════════════

def train(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[RandomForestClassifier, dict, dict]:
    """
    Train a RandomForestClassifier with an 80/20 stratified split.

    Stratified split ensures both classes (Eligible / Not Eligible) appear in
    roughly the same proportion in training and test sets — critical when the
    classes are imbalanced.

    Parameters
    ----------
    X : encoded feature matrix
    y : binary target vector (0 = Not Eligible, 1 = Eligible)

    Returns
    -------
    model      : fitted RandomForestClassifier
    split_data : {"X_train", "X_test", "y_train", "y_test"}
    train_info : sizes and hyperparams used
    """
    # ── Train / test split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_SEED,
        stratify     = y,          # preserve class ratio in both splits
    )

    # ── Fit the model ──────────────────────────────────────────────────────────
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    split_data = {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
    }
    train_info = {
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "n_features":    X.shape[1],
        "feature_names": list(X.columns),
        "hyperparams":   RF_PARAMS,
        "test_size":     TEST_SIZE,
    }
    return model, split_data, train_info


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluate
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_full: pd.DataFrame | None = None,
    y_full: pd.Series   | None = None,
) -> dict[str, Any]:
    """
    Compute all evaluation metrics used by the Streamlit ML page.

    Metrics returned
    ─────────────────
    accuracy          → % of correct predictions overall
    precision         → of all predicted Eligible, how many truly were?
    recall            → of all truly Eligible, how many did we catch?
    f1_score          → harmonic mean of precision and recall
    roc_auc           → area under ROC curve (1.0 = perfect, 0.5 = random)
    confusion_matrix  → [[TN, FP], [FN, TP]]
    classification_report → full per-class stats as a string
    cv_scores         → 5-fold cross-validation accuracy (uses X_full / y_full)
    feature_importance → {feature: importance_score} sorted descending
    """
    y_pred      = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Core metrics
    acc       = accuracy_score(y_test, y_pred)
    prec      = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec       = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_pred_proba)
    cm        = confusion_matrix(y_test, y_pred)
    clf_rpt   = classification_report(
        y_test, y_pred,
        target_names=["Not Eligible", "Eligible"],
        zero_division=0,
    )

    # 5-fold cross-validation (needs full dataset)
    cv_mean, cv_std = None, None
    if X_full is not None and y_full is not None:
        cv_scores = cross_val_score(model, X_full, y_full, cv=5, scoring="accuracy")
        cv_mean   = float(cv_scores.mean())
        cv_std    = float(cv_scores.std())

    # Feature importance from the Random Forest
    importances = dict(
        sorted(
            zip(X_test.columns, model.feature_importances_),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    return {
        "accuracy":               round(float(acc),     4),
        "precision":              round(float(prec),    4),
        "recall":                 round(float(rec),     4),
        "f1_score":               round(float(f1),      4),
        "roc_auc":                round(float(roc_auc), 4),
        "confusion_matrix":       cm.tolist(),
        "classification_report":  clf_rpt,
        "cv_mean":                round(cv_mean, 4) if cv_mean else None,
        "cv_std":                 round(cv_std,  4) if cv_std  else None,
        "feature_importance":     importances,
        "y_pred":                 y_pred.tolist(),
        "y_pred_proba":           y_pred_proba.tolist(),
        "y_test":                 y_test.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Persist
# ══════════════════════════════════════════════════════════════════════════════

def save_model(
    model:    RandomForestClassifier,
    encoders: dict[str, LabelEncoder],
    metrics:  dict[str, Any],
    train_info: dict[str, Any],
) -> None:
    """
    Persist model, encoders, and metadata to disk using joblib.

    joblib is preferred over pickle for sklearn objects because it is:
      • Faster for large numpy arrays (memory-mapped)
      • More reliable across Python / sklearn version changes
      • The official sklearn recommendation

    Files saved
    ───────────
    rf_model.pkl    — the fitted RandomForestClassifier object
    encoders.pkl    — dict of {column_name: fitted LabelEncoder}
    model_meta.json — human-readable JSON with all training stats
    """
    joblib.dump(model,    MODEL_PATH,    compress=3)
    joblib.dump(encoders, ENCODERS_PATH, compress=3)

    meta = {
        "trained_at":       datetime.now().isoformat(timespec="seconds"),
        "sklearn_version":  __import__("sklearn").__version__,
        "model_type":       "RandomForestClassifier",
        "hyperparameters":  {k: str(v) for k, v in RF_PARAMS.items()},
        "train_info":       {
            k: v for k, v in train_info.items()
            if k != "hyperparams"          # avoid duplicate
        },
        "metrics": {
            "accuracy":   metrics["accuracy"],
            "precision":  metrics["precision"],
            "recall":     metrics["recall"],
            "f1_score":   metrics["f1_score"],
            "roc_auc":    metrics["roc_auc"],
            "cv_mean":    metrics.get("cv_mean"),
            "cv_std":     metrics.get("cv_std"),
        },
        "feature_importance": metrics["feature_importance"],
    }
    with open(META_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)


def load_model() -> tuple[RandomForestClassifier, dict[str, LabelEncoder], dict]:
    """
    Load the saved model, encoders, and metadata from disk.

    Raises FileNotFoundError if the model has not been trained yet.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Please train the model first "
            "via the ML Training page."
        )
    model    = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    with open(META_PATH) as fh:
        meta = json.load(fh)
    return model, encoders, meta


def model_exists() -> bool:
    """Return True if a trained model artefact is present on disk."""
    return MODEL_PATH.exists()


# ══════════════════════════════════════════════════════════════════════════════
#  Predict (single applicant)
# ══════════════════════════════════════════════════════════════════════════════

def predict_single(
    model:    RandomForestClassifier,
    encoders: dict[str, LabelEncoder],
    features: dict,
) -> dict[str, Any]:
    """
    Predict eligibility for one applicant.

    Parameters
    ----------
    features : raw dict matching FEATURE_COLS keys:
        {
            "age": 34,
            "family_income": 95000.0,
            "family_members": 4,
            "employment_status": "Unemployed",
            "education_level": "Secondary",
            "disability_status": 0,
        }

    Returns
    -------
    {
        "label":       0 or 1,
        "verdict":     "Eligible" | "Not Eligible",
        "confidence":  0.0–1.0,  # probability for the predicted class
        "prob_eligible": float,
        "prob_not_eligible": float,
    }
    """
    row = dict(features)  # copy so we don't mutate caller's dict

    # Encode categorical columns using the SAVED encoders
    for col, le in encoders.items():
        val = row[col]
        known = set(le.classes_)
        if val not in known:
            val = le.classes_[0]   # fallback to first known class
        row[col] = int(le.transform([val])[0])

    # Build a 1-row DataFrame in the exact feature order the model was trained on
    from ml.preprocessor import FEATURE_COLS
    X = pd.DataFrame([row])[FEATURE_COLS]

    label       = int(model.predict(X)[0])
    proba       = model.predict_proba(X)[0]  # [p_not_eligible, p_eligible]
    confidence  = float(proba[label])

    return {
        "label":             label,
        "verdict":           "Eligible" if label == 1 else "Not Eligible",
        "confidence":        round(confidence, 4),
        "prob_eligible":     round(float(proba[1]), 4),
        "prob_not_eligible": round(float(proba[0]), 4),
    }
