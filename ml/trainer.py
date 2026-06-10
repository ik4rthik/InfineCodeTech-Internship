"""
ml/trainer.py
─────────────
Standalone training script — orchestrates the full ML pipeline end-to-end.

This file can be run directly from the terminal for a one-shot training run
without needing Streamlit:

    python ml/trainer.py

It can also be called programmatically from the Streamlit "ML Training" page
via the `run_pipeline()` function, which returns all artefacts needed by the UI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ── Project root on sys.path ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessor import preprocess, load_raw_data
from ml.model import train, evaluate, save_model


def run_pipeline(verbose: bool = True) -> dict[str, Any]:
    """
    Execute the full training pipeline and return all artefacts.

    Pipeline stages
    ───────────────
    1. Load raw data from SQLite
    2. Preprocess  (clean → dedup → impute → encode → split X/y)
    3. Train       (RandomForestClassifier, 80/20 stratified split)
    4. Evaluate    (accuracy, precision, recall, F1, ROC-AUC, CM, feature importance)
    5. Save        (rf_model.pkl, encoders.pkl, model_meta.json)

    Parameters
    ----------
    verbose : if True, prints progress to stdout

    Returns
    -------
    dict with keys:
        "preprocess_report"  — step-by-step preprocessing counts
        "train_info"         — split sizes, hyperparams
        "metrics"            — all evaluation metrics
        "model"              — fitted RandomForestClassifier
        "encoders"           — {col: LabelEncoder}
        "X"                  — full feature matrix (for cross-validation)
        "y"                  — full target vector
        "X_test"             — held-out test features
        "y_test"             — held-out test labels
    """

    def log(msg: str) -> None:
        if verbose:
            print(f"[trainer] {msg}")

    # ── Stage 1: Load ──────────────────────────────────────────────────────────
    log("Loading raw data from SQLite...")
    raw_df = load_raw_data()
    log(f"  Loaded {len(raw_df)} raw rows.")

    # ── Stage 2: Preprocess ────────────────────────────────────────────────────
    log("Running preprocessing pipeline...")
    X, y, encoders, prep_report = preprocess(df=raw_df)
    log(f"  Final dataset: {len(X)} rows x {X.shape[1]} features")
    log(f"  Class balance: {prep_report['class_balance']}")

    # ── Stage 3: Train ─────────────────────────────────────────────────────────
    log("Training RandomForestClassifier (200 trees, balanced weights)...")
    model, split_data, train_info = train(X, y)
    log(f"  Train set: {train_info['n_train']} samples")
    log(f"  Test  set: {train_info['n_test']} samples")

    # ── Stage 4: Evaluate ──────────────────────────────────────────────────────
    log("Evaluating model on held-out test set...")
    metrics = evaluate(
        model,
        split_data["X_test"],
        split_data["y_test"],
        X_full=X,
        y_full=y,
    )
    log(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
    log(f"  Precision : {metrics['precision']*100:.2f}%")
    log(f"  Recall    : {metrics['recall']*100:.2f}%")
    log(f"  F1 Score  : {metrics['f1_score']*100:.2f}%")
    log(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    if metrics.get("cv_mean"):
        log(f"  5-Fold CV : {metrics['cv_mean']*100:.2f}% (+/- {metrics['cv_std']*100:.2f}%)")

    # ── Stage 5: Save ──────────────────────────────────────────────────────────
    log("Saving model artefacts to ml/trained_models/...")
    save_model(model, encoders, metrics, train_info)
    log("  rf_model.pkl    saved")
    log("  encoders.pkl    saved")
    log("  model_meta.json saved")
    log("Training pipeline complete.")

    return {
        "preprocess_report": prep_report,
        "train_info":        train_info,
        "metrics":           metrics,
        "model":             model,
        "encoders":          encoders,
        "X":                 X,
        "y":                 y,
        "X_test":            split_data["X_test"],
        "y_test":            split_data["y_test"],
    }


if __name__ == "__main__":
    results = run_pipeline(verbose=True)
    print("\nClassification Report:")
    print(results["metrics"]["classification_report"])
