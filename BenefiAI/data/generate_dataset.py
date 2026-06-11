"""
data/generate_dataset.py
────────────────────────
Generates a realistic synthetic dataset of 300 NGO beneficiary applicants
and saves it to data/beneficiaries.csv.

Column definitions
──────────────────
id                 – Auto-incremented integer (1-based)
applicant_name     – Realistic Indian-ish name via Faker
age                – Integer, 18–70
family_income      – Annual household income in INR (5 000 – 6 00 000)
family_members     – Number of family members (1–10)
employment_status  – One of: Unemployed | Part-time | Full-time | Self-employed
education_level    – One of: No Formal | Primary | Secondary | Graduate | Post-Graduate
disability_status  – Yes | No
eligibility_status – Derived label: Eligible | Not Eligible
                     (rule-based heuristic; will be replaced by ML in Phase 2)

Eligibility heuristic
─────────────────────
A record is labelled "Eligible" when ANY two of the following hold:
  1. Annual income per family member < 30 000 INR
  2. Unemployed or Part-time employed
  3. Education level is No Formal or Primary
  4. Has a disability

Usage
─────
  python data/generate_dataset.py
"""

import os
import random
import numpy as np
import pandas as pd
from faker import Faker

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_IN")          # Indian locale for realistic names
Faker.seed(SEED)

# ── Constants ─────────────────────────────────────────────────────────────────
N_RECORDS = 300

EMPLOYMENT_STATUS = ["Unemployed", "Part-time", "Full-time", "Self-employed"]
EMPLOYMENT_WEIGHTS = [0.30, 0.25, 0.30, 0.15]   # skew toward vulnerable groups

EDUCATION_LEVEL = ["No Formal", "Primary", "Secondary", "Graduate", "Post-Graduate"]
EDUCATION_WEIGHTS = [0.10, 0.20, 0.35, 0.25, 0.10]

DISABILITY_OPTIONS = ["Yes", "No"]
DISABILITY_WEIGHTS = [0.15, 0.85]


# ── Eligibility rule ──────────────────────────────────────────────────────────
def determine_eligibility(
    family_income: float,
    family_members: int,
    employment_status: str,
    education_level: str,
    disability_status: str,
) -> str:
    """
    Rule-based eligibility heuristic.
    Returns 'Eligible' if at least 2 vulnerability flags are raised.
    """
    flags = 0

    # Flag 1 – Low per-capita income
    per_capita = family_income / max(family_members, 1)
    if per_capita < 30_000:
        flags += 1

    # Flag 2 – Weak employment
    if employment_status in ("Unemployed", "Part-time"):
        flags += 1

    # Flag 3 – Low education
    if education_level in ("No Formal", "Primary"):
        flags += 1

    # Flag 4 – Disability
    if disability_status == "Yes":
        flags += 1

    return "Eligible" if flags >= 2 else "Not Eligible"


# ── Generator ─────────────────────────────────────────────────────────────────
def generate_dataset(n: int = N_RECORDS) -> pd.DataFrame:
    records = []

    for i in range(1, n + 1):
        name              = fake.name()
        age               = random.randint(18, 70)
        family_members    = random.randint(1, 10)
        # Income distribution: log-normal gives realistic skew
        family_income     = round(
            np.random.lognormal(mean=10.8, sigma=0.7), 2
        )
        family_income     = max(5_000, min(family_income, 600_000))   # clamp

        employment_status = random.choices(
            EMPLOYMENT_STATUS, weights=EMPLOYMENT_WEIGHTS, k=1
        )[0]
        education_level   = random.choices(
            EDUCATION_LEVEL, weights=EDUCATION_WEIGHTS, k=1
        )[0]
        disability_status = random.choices(
            DISABILITY_OPTIONS, weights=DISABILITY_WEIGHTS, k=1
        )[0]
        eligibility_status = determine_eligibility(
            family_income, family_members,
            employment_status, education_level, disability_status,
        )

        records.append({
            "id":                i,
            "applicant_name":    name,
            "age":               age,
            "family_income":     family_income,
            "family_members":    family_members,
            "employment_status": employment_status,
            "education_level":   education_level,
            "disability_status": disability_status,
            "eligibility_status": eligibility_status,
        })

    df = pd.DataFrame(records)
    return df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    output_dir  = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "beneficiaries.csv")

    df = generate_dataset()

    eligible_count     = (df["eligibility_status"] == "Eligible").sum()
    not_eligible_count = (df["eligibility_status"] == "Not Eligible").sum()

    df.to_csv(output_path, index=False)

    print("-" * 55)
    print("  BenefiAI - Dataset Generation Complete")
    print("-" * 55)
    print(f"  Total records  : {len(df)}")
    print(f"  Eligible       : {eligible_count} ({eligible_count/len(df)*100:.1f}%)")
    print(f"  Not Eligible   : {not_eligible_count} ({not_eligible_count/len(df)*100:.1f}%)")
    print(f"  Saved to       : {output_path}")
    print("-" * 55)
