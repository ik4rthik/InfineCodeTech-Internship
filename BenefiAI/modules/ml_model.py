"""
modules/ml_model.py
═══════════════════
BenefiAI – Machine Learning Pipeline

End-to-end implementation for NGO Beneficiary Eligibility Prediction.

PIPELINE OVERVIEW
─────────────────
  Step 1 │ load_raw_data()          Load DataFrame from SQLite
  Step 2 │ clean(df)                Handle nulls, duplicates, bad values
  Step 3 │ encode(df)               Label-encode all categorical columns
  Step 4 │ split(X, y)              Stratified 80/20 train-test split
  Step 5 │ train(X_train, y_train)  Fit RandomForestClassifier
  Step 6 │ evaluate(...)            Compute Accuracy / Precision / Recall /
          │                          F1 / ROC-AUC + Confusion Matrix
  Step 7 │ save_model(bundle)       Persist model + encoders via joblib
  Step 8 │ load_model()             Reload for inference
  Step 9 │ predict_single(...)      Single-applicant probability prediction
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(_HERE, "..", "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "eligibility_model.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Column definitions ────────────────────────────────────────────────────────
FEATURE_COLS = [
    "age",
    "family_income",
    "family_members",
    "employment_status",   # categorical → encoded
    "education_level",     # categorical → encoded
    "disability_status",   # categorical → encoded
]
TARGET_COL   = "eligibility_status"
CAT_COLS     = ["employment_status", "education_level", "disability_status"]

# Fixed category vocabularies (must stay consistent between train & predict)
CATEGORY_VOCAB = {
    "employment_status": ["Unemployed", "Part-time", "Full-time", "Self-employed"],
    "education_level":   ["No Formal", "Primary", "Secondary", "Graduate", "Post-Graduate"],
    "disability_status": ["No", "Yes"],
    TARGET_COL:          ["Not Eligible", "Eligible"],   # 0 / 1
}

# Random forest hyper-parameters
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,       # grow full trees; forest bagging handles overfitting
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",   # handles class imbalance automatically
    random_state=42,
    n_jobs=-1,
)

TEST_SIZE   = 0.20   # 80 % train / 20 % test
RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 – LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_data() -> pd.DataFrame:
    """
    Load the beneficiaries table from SQLite into a Pandas DataFrame.

    Why SQLite and not the CSV?
    ───────────────────────────
    The database reflects any CRUD edits made through the dashboard, so
    training on DB data means the model always sees the most current records.
    """
    from database.crud import fetch_all   # late import avoids circular deps
    df = fetch_all()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 – DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning pipeline.  Returns a clean copy of the DataFrame.

    Operations (in order)
    ─────────────────────
    1. Drop exact duplicate rows.
    2. Strip leading/trailing whitespace from all string columns.
    3. Standardise case for categorical columns (Title Case).
    4. Drop rows with NaN in any required column.
    5. Remove physically impossible values (age < 1, income < 0, members < 1).
    6. Clamp outlier income values to [1_000, 10_000_000] INR.
    7. Reset index.

    Returns
    -------
    pd.DataFrame – cleaned copy
    """
    required_cols = FEATURE_COLS + [TARGET_COL]
    df = df.copy()

    # 1. Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    removed_dups = before - len(df)

    # 2. Strip whitespace in object columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # 3. Title-case ONLY simple yes/no and eligibility columns.
    #    Do NOT title-case employment_status ("Full-time" → "Full-Time" breaks vocab)
    #    or education_level ("No Formal" → "No Formal" is fine, but "Post-Graduate"
    #    would survive anyway; safer to leave them as-is and just strip).
    for col in ["disability_status", TARGET_COL]:
        if col in df.columns:
            df[col] = df[col].str.title()

    # Re-map "Not Eligible" in case it was stored differently
    df[TARGET_COL] = df[TARGET_COL].replace(
        {"Not Eligible": "Not Eligible", "Eligible": "Eligible"}
    )

    # 4. Drop rows with any NaN in required columns
    before = len(df)
    df = df.dropna(subset=required_cols)
    removed_nulls = before - len(df)

    # 5. Remove impossible values
    df = df[df["age"].between(1, 120)]
    df = df[df["family_income"] >= 0]
    df = df[df["family_members"] >= 1]

    # 6. Clamp income outliers
    df["family_income"] = df["family_income"].clip(lower=1_000, upper=10_000_000)

    # 7. Reset index
    df = df.reset_index(drop=True)

    cleaning_log = {
        "duplicates_removed": removed_dups,
        "nulls_removed":      removed_nulls,
        "final_rows":         len(df),
    }
    return df, cleaning_log


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 – LABEL ENCODING
# ══════════════════════════════════════════════════════════════════════════════

def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode all categorical columns using fixed vocabularies.

    Why fixed vocabularies (not fit-on-data)?
    ──────────────────────────────────────────
    If we fit encoders on training data only, any category unseen at train time
    would cause a KeyError at inference time.  Fixing the vocab up-front means
    the mapping is always consistent — e.g. "Yes" → 1, "No" → 0 regardless of
    whether the training split happened to contain both.

    Returns
    ───────
    (encoded_df, encoders_dict)
      encoded_df   – DataFrame with integer-coded categoricals
      encoders_dict – {col_name: fitted LabelEncoder}
    """
    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in CAT_COLS + [TARGET_COL]:
        le = LabelEncoder()
        le.classes_ = np.array(CATEGORY_VOCAB[col])   # fix the vocab
        df[col] = le.transform(df[col])
        encoders[col] = le

    return df, encoders


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 – TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def split(X: pd.DataFrame, y: pd.Series):
    """
    Stratified 80/20 train-test split.

    Stratification ensures both splits preserve the class ratio
    (Eligible vs Not Eligible), which is important when classes are imbalanced.

    Returns
    ───────
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 – TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Fit a RandomForestClassifier on the training data.

    Why Random Forest?
    ──────────────────
    • Handles mixed feature types well (numerics + encoded categoricals).
    • Robust to outliers — decision tree splits are threshold-based.
    • class_weight='balanced' internally compensates for class imbalance.
    • Provides feature_importances_ for explainability.
    • No need to scale features (unlike SVMs or logistic regression).
    • 200 trees with bagging reduces variance significantly.

    Returns
    ───────
    Fitted RandomForestClassifier
    """
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)
    return clf


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 – EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Metrics explained
    ─────────────────
    Accuracy   – Overall % of correct predictions.
                 Misleading when classes are imbalanced → use with caution.
    Precision  – Of all predicted "Eligible", what fraction truly is?
                 High precision = fewer false positives (less wasted aid).
    Recall     – Of all truly Eligible, what fraction did we catch?
                 High recall = fewer missed beneficiaries (critical for NGOs).
    F1 Score   – Harmonic mean of Precision & Recall.
                 Best single metric when both FP and FN matter.
    ROC-AUC    – Area under the ROC curve.  0.5 = random, 1.0 = perfect.
                 Measures discrimination ability across all thresholds.
    CV Score   – 5-fold cross-validation accuracy (more reliable than
                 a single train-test split).

    Confusion Matrix
    ────────────────
                 Predicted Not-Eligible  |  Predicted Eligible
    Actual Not-Elig       TN            |        FP
    Actual Eligible       FN            |        TP

    Returns
    ───────
    dict with all metrics + confusion matrix array + classification report
    """
    y_pred      = model.predict(X_test)
    y_proba     = model.predict_proba(X_test)[:, 1]   # P(Eligible)

    # 5-fold CV on training data
    cv_scores   = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    metrics = {
        "accuracy":          round(accuracy_score(y_test, y_pred),            4),
        "precision":         round(precision_score(y_test, y_pred, pos_label=1), 4),
        "recall":            round(recall_score(y_test, y_pred, pos_label=1),    4),
        "f1_score":          round(f1_score(y_test, y_pred, pos_label=1),        4),
        "roc_auc":           round(roc_auc_score(y_test, y_proba),               4),
        "cv_mean_accuracy":  round(cv_scores.mean(),                             4),
        "cv_std":            round(cv_scores.std(),                              4),
        "confusion_matrix":  confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=["Not Eligible", "Eligible"],
            output_dict=True,
        ),
        "train_accuracy":    round(accuracy_score(y_train, model.predict(X_train)), 4),
        "test_size":         len(y_test),
        "train_size":        len(y_train),
    }
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 – SAVE / LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def save_model(bundle: dict, path: str = MODEL_PATH) -> None:
    """
    Persist the model bundle to disk using joblib.

    What is saved (the bundle dict)
    ───────────────────────────────
    {
      "model":    trained RandomForestClassifier,
      "encoders": {col: LabelEncoder, …},
      "features": list of feature column names,
      "metrics":  evaluation metrics dict,
    }

    Why joblib?
    ───────────
    joblib is faster than pickle for large NumPy arrays (the forest's
    decision trees are stored as arrays internally).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)


def load_model(path: str = MODEL_PATH) -> dict | None:
    """
    Load the model bundle from disk.

    Returns None if no model file exists yet (first run).
    """
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def model_exists(path: str = MODEL_PATH) -> bool:
    """Return True if a trained model file exists on disk."""
    return os.path.exists(path)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 – FULL PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline() -> dict:
    """
    Execute Steps 1-7 end-to-end and return a result dict.

    Returns
    ───────
    {
      "success":      bool,
      "metrics":      dict,
      "cleaning_log": dict,
      "features":     list,
      "bundle":       dict,   ← also saved to disk
      "error":        str,    ← only present on failure
    }
    """
    try:
        # Step 1: Load
        raw_df = load_raw_data()
        if len(raw_df) < 20:
            return {"success": False, "error": "Need at least 20 records to train."}

        # Step 2: Clean
        clean_df, cleaning_log = clean(raw_df)
        if len(clean_df) < 10:
            return {"success": False, "error": "Too few records after cleaning."}

        # Step 3: Encode
        encoded_df, encoders = encode(clean_df)

        # Step 4: Split features / target
        X = encoded_df[FEATURE_COLS]
        y = encoded_df[TARGET_COL]
        X_train, X_test, y_train, y_test = split(X, y)

        # Step 5: Train
        model = train(X_train, y_train)

        # Step 6: Evaluate
        metrics = evaluate(model, X_train, X_test, y_train, y_test)

        # Step 7: Save
        bundle = {
            "model":    model,
            "encoders": encoders,
            "features": FEATURE_COLS,
            "metrics":  metrics,
        }
        save_model(bundle)

        return {
            "success":      True,
            "metrics":      metrics,
            "cleaning_log": cleaning_log,
            "features":     FEATURE_COLS,
            "bundle":       bundle,
        }

    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9 – SINGLE-APPLICANT INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def predict_single(bundle: dict, applicant: dict) -> dict:
    """
    Predict eligibility for one applicant using the trained model.

    Parameters
    ──────────
    bundle    : dict returned by load_model()
    applicant : dict with keys:
                  age, family_income, family_members,
                  employment_status, education_level, disability_status

    Returns
    ───────
    {
      "label":      "Eligible" | "Not Eligible",
      "confidence": float  0.0–1.0  (probability of being Eligible),
      "flag_detail": dict  showing which vulnerability flags fired
    }
    """
    model    = bundle["model"]
    encoders = bundle["encoders"]

    # Build a one-row DataFrame
    row = {col: [applicant[col]] for col in FEATURE_COLS}
    df  = pd.DataFrame(row)

    # Encode categorical columns
    for col in CAT_COLS:
        df[col] = encoders[col].transform(df[col])

    # Predict
    label_encoded = model.predict(df)[0]
    proba         = model.predict_proba(df)[0]        # [P(Not Elig), P(Elig)]
    confidence    = float(proba[1])                   # P(Eligible)
    label         = encoders[TARGET_COL].inverse_transform([label_encoded])[0]

    # Compute vulnerability flags for transparency
    per_capita = applicant["family_income"] / max(applicant["family_members"], 1)
    flags = {
        "low_per_capita_income": per_capita < 30_000,
        "weak_employment":       applicant["employment_status"] in ("Unemployed", "Part-time"),
        "low_education":         applicant["education_level"] in ("No Formal", "Primary"),
        "has_disability":        applicant["disability_status"] == "Yes",
    }
    flags["total_flags"] = sum(flags.values())

    return {
        "label":      label,
        "confidence": round(confidence, 4),
        "flags":      flags,
        "per_capita_income": round(per_capita, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def get_feature_importance(bundle: dict) -> pd.DataFrame:
    """
    Return a sorted DataFrame of feature importances from the forest.

    Feature importance in a Random Forest = mean decrease in Gini impurity
    across all trees.  Higher = more predictive.
    """
    model    = bundle["model"]
    features = bundle["features"]
    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature":    features,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    df["importance_pct"] = (df["importance"] / df["importance"].sum() * 100).round(2)
    return df
