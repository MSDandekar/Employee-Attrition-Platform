"""
03_generate_raw_dataset.py
===========================
Builds the "realistic raw" Employee Attrition dataset for Experiments 1 & 2.

STRATEGY (matches what we discussed):
  1. IBM HR Analytics = our BASE population (1,470 rows, kept complete - no
     row-stacking with HRDataset_v14, no structural holes).
  2. HRDataset_v14 is used only to donate EMPIRICAL DISTRIBUTIONS for three
     concepts IBM lacks (race/ethnicity, recruitment source, manager
     structure) - so every IBM row gets a value, nothing is left blank.
  3. We then deliberately inject documented, reproducible data-quality
     issues onto this enriched base, modeled on messiness patterns
     genuinely observed in HRDataset_v14 (mixed date formats, whitespace,
     free-text exit reasons, etc).
  4. Every single injection is written to `injection_log` - this is your
     ANSWER KEY. It lets you (and your professor) verify your Experiment 2
     cleaning pipeline actually recovers the right values, since you know
     ground truth. Keep the log OUT of the file you hand to your cleaning
     pipeline - that's the "raw" file. The log is your private verification
     artifact.

Everything is seeded (RANDOM_SEED) so this is 100% reproducible - running
this script twice produces byte-identical output.
"""

import pandas as pd
import numpy as np

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

REFERENCE_DATE = pd.Timestamp("2020-01-01")  # our fictional "company snapshot" date

# ---------------------------------------------------------------------------
# 0. LOAD RAW FILES
# ---------------------------------------------------------------------------

def load_data():
    df_ibm = pd.read_csv(r"C:\Users\Student.DESKTOP-2MHLOHF.003\Downloads\A716_ADS\data\raw\IBM_HR_Attrition.csv")
    df_hrd = pd.read_csv(r"C:\Users\Student.DESKTOP-2MHLOHF.003\Downloads\A716_ADS\data\raw\HRDataset_v14.csv")
    return df_ibm, df_hrd


# ---------------------------------------------------------------------------
# 1. ENRICHMENT: donate concepts (not rows) from HRDataset_v14
# ---------------------------------------------------------------------------
# WHY THIS IS NOT A FAKE JOIN: we are not saying "this IBM employee IS this
# HRDataset employee." We are saying "here is the empirical PROBABILITY
# DISTRIBUTION of race/recruitment-source/etc observed in a real HR dataset -
# use it to responsibly simulate these fields for our base population."
# This is disclosed, seeded, and documented - the opposite of quietly
# fabricating data.

def enrich_with_hrd_distributions(df_ibm: pd.DataFrame, df_hrd: pd.DataFrame) -> pd.DataFrame:
    df = df_ibm.copy()
    n = len(df)

    # --- race_ethnicity: sample from HRDataset_v14's real category frequencies ---
    race_probs = df_hrd["RaceDesc"].value_counts(normalize=True)
    df["race_ethnicity"] = rng.choice(race_probs.index, size=n, p=race_probs.values)

    # --- recruitment_source: same idea ---
    source_probs = df_hrd["RecruitmentSource"].value_counts(normalize=True)
    df["recruitment_source"] = rng.choice(source_probs.index, size=n, p=source_probs.values)

    # --- manager_id: synthesize a plausible manager hierarchy, grouped by
    # department (managers realistically only manage within their own dept).
    # Roughly 1 manager per 8 employees, minimum 2 managers per department.
    manager_lookup = {}
    df["manager_id"] = ""
    for dept, group in df.groupby("Department"):
        n_managers = max(2, len(group) // 8)
        manager_ids = [f"MGR_{dept[:3].upper()}_{i:02d}" for i in range(1, n_managers + 1)]
        assigned = rng.choice(manager_ids, size=len(group))
        df.loc[group.index, "manager_id"] = assigned

    return df


# ---------------------------------------------------------------------------
# 2. RECONSTRUCT A HIRE DATE (IBM only stores YearsAtCompany, not a real date)
# ---------------------------------------------------------------------------

def reconstruct_hire_date(df: pd.DataFrame) -> pd.Series:
    # Convert integer tenure into an actual calendar date relative to our
    # fixed reference/snapshot date, with a random day-of-year jitter so
    # hire dates aren't suspiciously all on Jan 1st.
    days_employed = (df["YearsAtCompany"] * 365.25).round().astype(int)
    jitter_days = rng.integers(0, 365, size=len(df))
    hire_date = REFERENCE_DATE - pd.to_timedelta(days_employed, unit="D") \
                + pd.to_timedelta(jitter_days, unit="D")
    return hire_date


# ---------------------------------------------------------------------------
# 3. INJECTION LOG helper
# ---------------------------------------------------------------------------
# Every injection function appends rows here: which row, which column, what
# TYPE of issue, and the ORIGINAL (clean) value before we corrupted it.
# This is what makes the "raw" file gradeable/verifiable rather than just
# messy for messy's sake.

injection_log_rows = []

def log_injection(row_index, column, issue_type, original_value):
    injection_log_rows.append({
        "row_index": row_index,
        "column": column,
        "issue_type": issue_type,
        "original_value": original_value,
    })


# ---------------------------------------------------------------------------
# 4. MISSING VALUES - three distinct mechanisms (MCAR, MAR, MNAR)
# ---------------------------------------------------------------------------

def inject_missingness(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- MCAR: DistanceFromHome missing completely at random (~4% of rows) ---
    mcar_idx = df.sample(frac=0.04, random_state=RANDOM_SEED).index
    for idx in mcar_idx:
        log_injection(idx, "DistanceFromHome", "MCAR", df.loc[idx, "DistanceFromHome"])
    df.loc[mcar_idx, "DistanceFromHome"] = np.nan

    # --- MAR: MonthlyIncome missing MORE OFTEN for newer hires (YearsAtCompany
    # <= 1), mimicking payroll records lagging behind for recent starters.
    new_hire_idx = df[df["YearsAtCompany"] <= 1].index
    mar_idx = pd.Index(rng.choice(new_hire_idx, size=int(len(new_hire_idx) * 0.35), replace=False)) \
        if len(new_hire_idx) > 0 else pd.Index([])
    for idx in mar_idx:
        log_injection(idx, "MonthlyIncome", "MAR (conditional on new-hire status)", df.loc[idx, "MonthlyIncome"])
    df.loc[mar_idx, "MonthlyIncome"] = np.nan

    # --- MNAR: PerformanceRating more likely missing for employees who left
    # (Attrition == 'Yes'), mimicking incomplete exit paperwork.
    leavers_idx = df[df["Attrition"] == "Yes"].index
    mnar_idx = pd.Index(rng.choice(leavers_idx, size=int(len(leavers_idx) * 0.25), replace=False)) \
        if len(leavers_idx) > 0 else pd.Index([])
    for idx in mnar_idx:
        log_injection(idx, "PerformanceRating", "MNAR (conditional on Attrition=Yes)", df.loc[idx, "PerformanceRating"])
    df.loc[mnar_idx, "PerformanceRating"] = np.nan

    return df


# ---------------------------------------------------------------------------
# 5. CATEGORICAL INCONSISTENCY - case, whitespace, synonyms
# ---------------------------------------------------------------------------

def inject_categorical_inconsistency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Department: mixed case + whitespace + a synonym swap for R&D
    dept_variants = {
        "Sales": ["Sales", "sales", "SALES", " Sales"],
        "Research & Development": ["Research & Development", "R&D", "research & development", "Research and Development "],
        "Human Resources": ["Human Resources", "human resources", "HR", "Human Resources "],
    }
    dept_idx = df.sample(frac=0.15, random_state=RANDOM_SEED + 1).index
    for idx in dept_idx:
        original = df.loc[idx, "Department"]
        variants = dept_variants.get(original)
        if variants:
            new_val = rng.choice(variants)
            if new_val != original:
                log_injection(idx, "Department", "categorical inconsistency", original)
                df.loc[idx, "Department"] = new_val

    # OverTime: swap 'Yes'/'No' for 'Y'/'N' on a subset (abbreviation drift)
    ot_idx = df.sample(frac=0.10, random_state=RANDOM_SEED + 2).index
    ot_map = {"Yes": "Y", "No": "N"}
    for idx in ot_idx:
        original = df.loc[idx, "OverTime"]
        log_injection(idx, "OverTime", "categorical inconsistency (abbreviation)", original)
        df.loc[idx, "OverTime"] = ot_map.get(original, original)

    # Gender: trailing whitespace on a subset
    gender_idx = df.sample(frac=0.08, random_state=RANDOM_SEED + 3).index
    for idx in gender_idx:
        original = df.loc[idx, "Gender"]
        log_injection(idx, "Gender", "categorical inconsistency (whitespace)", original)
        df.loc[idx, "Gender"] = original + " "

    return df


# ---------------------------------------------------------------------------
# 6. MIXED DATE FORMATS + a few invalid dates
# ---------------------------------------------------------------------------

def inject_date_messiness(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hire_dates = reconstruct_hire_date(df)

    formats = [
        lambda d: d.strftime("%m/%d/%Y"),
        lambda d: d.strftime("%d-%b-%Y"),
        lambda d: d.strftime("%Y-%m-%d"),
        lambda d: d.strftime("%d/%m/%y"),
    ]
    format_choice = rng.integers(0, len(formats), size=len(df))
    hire_date_str = pd.Series(
        [formats[fmt_i](d) for fmt_i, d in zip(format_choice, hire_dates)],
        index=df.index,
    )

    # Inject a small number of outright invalid date strings (~1%)
    invalid_idx = df.sample(frac=0.01, random_state=RANDOM_SEED + 4).index
    for idx in invalid_idx:
        log_injection(idx, "HireDate", "invalid date string", hire_date_str.loc[idx])
        hire_date_str.loc[idx] = "00/00/0000"

    df["HireDate"] = hire_date_str
    return df


# ---------------------------------------------------------------------------
# 7. DUPLICATE AND NEAR-DUPLICATE ROWS
# ---------------------------------------------------------------------------

def inject_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Exact duplicates (~1% of rows re-appended as-is)
    exact_dupe_sample = df.sample(frac=0.01, random_state=RANDOM_SEED + 5)
    for idx in exact_dupe_sample.index:
        log_injection(idx, "ALL", "exact duplicate row inserted", "duplicated as-is")

    # Near-duplicates (~1% of rows re-appended with one field slightly off,
    # simulating a second, slightly-different data-entry event for the
    # same employee)
    near_dupe_sample = df.sample(frac=0.01, random_state=RANDOM_SEED + 6).copy()
    for idx in near_dupe_sample.index:
        original_income = df.loc[idx, "MonthlyIncome"]
        log_injection(idx, "MonthlyIncome", "near-duplicate row (income re-entry drift)", original_income)
    near_dupe_sample["MonthlyIncome"] = near_dupe_sample["MonthlyIncome"] * rng.uniform(0.98, 1.02, size=len(near_dupe_sample))
    near_dupe_sample["MonthlyIncome"] = near_dupe_sample["MonthlyIncome"].round().astype(int)

    combined = pd.concat([df, exact_dupe_sample, near_dupe_sample], ignore_index=False)
    return combined


# ---------------------------------------------------------------------------
# 8. OUTLIERS AND LOGICALLY IMPOSSIBLE VALUES
# ---------------------------------------------------------------------------

def inject_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Statistical outliers in MonthlyIncome (~0.5% of rows)
    income_outlier_idx = df.sample(frac=0.005, random_state=RANDOM_SEED + 7).index
    for idx in income_outlier_idx:
        log_injection(idx, "MonthlyIncome", "statistical outlier", df.loc[idx, "MonthlyIncome"])
    df.loc[income_outlier_idx, "MonthlyIncome"] = df.loc[income_outlier_idx, "MonthlyIncome"] * 15

    # Logically impossible ages (~0.3% of rows) - not just statistical
    # outliers, but values that break domain logic (Age can't be 5 or 130
    # for a working adult)
    age_outlier_idx = df.sample(frac=0.003, random_state=RANDOM_SEED + 8).index
    impossible_ages = rng.choice([5, 12, 130, 145], size=len(age_outlier_idx))
    for idx, bad_age in zip(age_outlier_idx, impossible_ages):
        log_injection(idx, "Age", "logically impossible value", df.loc[idx, "Age"])
    df.loc[age_outlier_idx, "Age"] = impossible_ages

    # Referential-integrity break: YearsAtCompany > (Age - 16), i.e. tenure
    # implies the person started working before age 16 (~0.3% of rows)
    integrity_idx = df.sample(frac=0.003, random_state=RANDOM_SEED + 9).index
    for idx in integrity_idx:
        log_injection(idx, "YearsAtCompany", "referential integrity violation (tenure > plausible working age)", df.loc[idx, "YearsAtCompany"])
    df.loc[integrity_idx, "YearsAtCompany"] = df.loc[integrity_idx, "Age"] + 5  # impossible: tenure exceeds age

    return df


# ---------------------------------------------------------------------------
# 9. UNIT MISMATCH - a subset of incomes stored as ANNUAL instead of MONTHLY
# ---------------------------------------------------------------------------

def inject_unit_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    unit_idx = df.sample(frac=0.03, random_state=RANDOM_SEED + 10).index
    for idx in unit_idx:
        log_injection(idx, "MonthlyIncome", "unit mismatch (stored as annual, not monthly)", df.loc[idx, "MonthlyIncome"])
    df.loc[unit_idx, "MonthlyIncome"] = df.loc[unit_idx, "MonthlyIncome"] * 12
    return df


# ---------------------------------------------------------------------------
# 10. FREE-TEXT EXIT NOTES (only for employees who left)
# ---------------------------------------------------------------------------
# Short, original phrases (not copied from any source) covering both
# voluntary and involuntary framings, so you can practice the exact
# .str.contains() voluntary/involuntary exclusion logic from your schema
# mapping work on your OWN dataset.

VOLUNTARY_NOTES = [
    "accepted a role elsewhere",
    "relocating out of state",
    "returning to full-time education",
    "pursuing a career change",
    "compensation not competitive",
]
INVOLUNTARY_NOTES = [
    "position eliminated - restructuring",
    "department downsizing",
    "role made redundant",
]

def add_termination_notes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TerminationNotes"] = ""
    leaver_idx = df[df["Attrition"] == "Yes"].index
    # ~80% voluntary framing, ~20% involuntary - matches the real-world
    # skew you'd expect, and gives you both categories to filter on later.
    is_involuntary = rng.random(len(leaver_idx)) < 0.20
    for idx, involuntary in zip(leaver_idx, is_involuntary):
        pool = INVOLUNTARY_NOTES if involuntary else VOLUNTARY_NOTES
        df.loc[idx, "TerminationNotes"] = rng.choice(pool)
    return df


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def build_raw_dataset():
    df_ibm, df_hrd = load_data()

    df = enrich_with_hrd_distributions(df_ibm, df_hrd)
    df = inject_missingness(df)
    df = inject_categorical_inconsistency(df)
    df = inject_date_messiness(df)
    df = inject_outliers(df)
    df = inject_unit_mismatch(df)
    df = add_termination_notes(df)
    df = inject_duplicates(df)  # duplicates added LAST so index-based logging above stays clean

    injection_log = pd.DataFrame(injection_log_rows)
    return df, injection_log


if __name__ == "__main__":
    raw_df, log_df = build_raw_dataset()

    raw_df.to_csv(r"C:\Users\Student.DESKTOP-2MHLOHF.003\Downloads\A716_ADS\data\processed\employee_attrition_raw.csv", index=False)
    log_df.to_csv(r"C:\Users\Student.DESKTOP-2MHLOHF.003\Downloads\A716_ADS\data\processed\injection_log.csv", index=False)

    print("Final raw dataset shape:", raw_df.shape)
    print("Injection log entries:", len(log_df))
    print("\nIssue type breakdown:")
    print(log_df["issue_type"].value_counts())
    print("\nMissing values per column (top 10):")
    print(raw_df.isna().sum().sort_values(ascending=False).head(10))
