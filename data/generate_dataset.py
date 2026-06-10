"""
data/generate_dataset.py
────────────────────────
Generates a realistic synthetic dataset of 300 NGO beneficiary applicants
and loads them into the SQLite database AND saves a CSV copy.

Design notes
  • Uses the Faker library for realistic names.
  • Employment, education, disability and income are correlated to mimic
    real-world patterns so the later ML model can learn meaningful signals.
  • Eligibility rule (deterministic + noise):
      Eligible if  family_income < INCOME_THRESHOLD
               AND (disability_status == 1 OR family_members >= 4
                    OR employment_status in {'Unemployed', 'Student'})
      ~10 % label noise is injected so the dataset isn't perfectly linearly
      separable and the classifier has something real to learn.

Usage
    python data/generate_dataset.py          # writes DB + CSV
    python data/generate_dataset.py --csv-only  # skip DB write
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ── Allow running from any working directory ───────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db_setup import get_connection, initialise_db

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_IN")   # Indian English locale for realistic names
Faker.seed(SEED)

# ── Constants ──────────────────────────────────────────────────────────────────
N_RECORDS = 300
INCOME_THRESHOLD = 180_000          # annual family income in INR
OUTPUT_CSV = PROJECT_ROOT / "data" / "beneficiaries.csv"

EMPLOYMENT_OPTIONS = ["Employed", "Unemployed", "Self-Employed", "Student", "Retired"]
EDUCATION_OPTIONS  = [
    "No Formal", "Primary", "Secondary",
    "Higher Secondary", "Graduate", "Post-Graduate",
]

# Weighted distributions to reflect typical beneficiary pools
EMPLOYMENT_WEIGHTS = [0.20, 0.35, 0.18, 0.15, 0.12]
EDUCATION_WEIGHTS  = [0.10, 0.18, 0.25, 0.22, 0.17, 0.08]


# ── Helpers ────────────────────────────────────────────────────────────────────
def _sample_income(employment: str, education: str) -> float:
    """
    Sample a plausible annual family income (INR) correlated with
    employment and education status.
    """
    base_map = {
        "Employed":      (220_000, 120_000),
        "Self-Employed": (180_000, 100_000),
        "Retired":       (130_000,  70_000),
        "Unemployed":    ( 80_000,  50_000),
        "Student":       ( 60_000,  40_000),
    }
    edu_multiplier = {
        "No Formal": 0.6, "Primary": 0.7, "Secondary": 0.85,
        "Higher Secondary": 1.0, "Graduate": 1.25, "Post-Graduate": 1.50,
    }
    mean, std = base_map[employment]
    income = np.random.normal(mean * edu_multiplier[education], std)
    return max(0.0, round(income, 2))


def _determine_eligibility(row: dict, noise: bool = True) -> int:
    """
    Rule-based eligibility label with optional ~10 % random flip (label noise).
    """
    low_income   = row["family_income"] < INCOME_THRESHOLD
    vulnerable   = (
        row["disability_status"] == 1
        or row["family_members"] >= 4
        or row["employment_status"] in {"Unemployed", "Student"}
    )
    eligible = int(low_income and vulnerable)

    if noise and random.random() < 0.10:   # 10 % label flip
        eligible = 1 - eligible

    return eligible


# ── Generator ──────────────────────────────────────────────────────────────────
def generate_records(n: int = N_RECORDS) -> pd.DataFrame:
    """Return a DataFrame with `n` synthetic beneficiary records."""
    records = []
    for i in range(1, n + 1):
        employment = random.choices(EMPLOYMENT_OPTIONS, EMPLOYMENT_WEIGHTS)[0]
        education  = random.choices(EDUCATION_OPTIONS,  EDUCATION_WEIGHTS )[0]

        # Age: working-age adults skewed distribution
        if employment == "Student":
            age = int(np.clip(np.random.normal(21, 4), 15, 35))
        elif employment == "Retired":
            age = int(np.clip(np.random.normal(62, 7), 50, 85))
        else:
            age = int(np.clip(np.random.normal(38, 12), 18, 75))

        family_income  = _sample_income(employment, education)
        family_members = int(np.clip(np.random.poisson(4), 1, 12))

        # Disability more common in lower-income, older groups
        dis_prob = 0.20 if (age > 50 or family_income < 100_000) else 0.08
        disability_status = int(random.random() < dis_prob)

        row = {
            "id":                 i,
            "applicant_name":     fake.name(),
            "age":                age,
            "family_income":      family_income,
            "family_members":     family_members,
            "employment_status":  employment,
            "education_level":    education,
            "disability_status":  disability_status,
        }
        row["eligibility_status"] = _determine_eligibility(row)
        records.append(row)

    df = pd.DataFrame(records)
    return df


# ── DB loader ──────────────────────────────────────────────────────────────────
def load_to_db(df: pd.DataFrame) -> None:
    """Insert records into the beneficiaries table (clears existing rows first)."""
    initialise_db()                         # ensure schema exists
    with get_connection() as conn:
        conn.execute("DELETE FROM beneficiaries;")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='beneficiaries';")

        insert_sql = """
            INSERT INTO beneficiaries
                (applicant_name, age, family_income, family_members,
                 employment_status, education_level, disability_status, eligibility_status)
            VALUES
                (:applicant_name, :age, :family_income, :family_members,
                 :employment_status, :education_level, :disability_status, :eligibility_status)
        """
        # Drop the synthetic 'id' column – SQLite will auto-assign it
        rows = df.drop(columns=["id"]).to_dict(orient="records")
        conn.executemany(insert_sql, rows)
        conn.commit()
    print(f"[generate_dataset] OK  {len(df)} records loaded into SQLite.")


# ── CSV writer ─────────────────────────────────────────────────────────────────
def save_csv(df: pd.DataFrame) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[generate_dataset] OK  CSV saved -> {OUTPUT_CSV}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(csv_only: bool = False) -> None:
    print(f"[generate_dataset] Generating {N_RECORDS} synthetic records...")
    df = generate_records(N_RECORDS)

    # Quick sanity checks
    assert len(df) == N_RECORDS,             "Record count mismatch"
    assert df.isnull().sum().sum() == 0,     "Unexpected nulls in dataset"
    assert set(df["eligibility_status"].unique()).issubset({0, 1}), "Bad labels"

    eligible_pct = df["eligibility_status"].mean() * 100
    print(f"[generate_dataset]    Eligible   : {eligible_pct:.1f} %")
    print(f"[generate_dataset]    Ineligible : {100 - eligible_pct:.1f} %")

    save_csv(df)
    if not csv_only:
        load_to_db(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BenefiAI synthetic dataset")
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Write CSV only; skip database insertion",
    )
    args = parser.parse_args()
    main(csv_only=args.csv_only)
