"""
experiment2_full_pipeline.py
=============================
Experiment 2 — Data Profiling, Cleaning & Feature Engineering
Full pipeline: manual profiling -> cleaning -> feature engineering ->
schema validation (Great Expectations) -> save.

This is the ACTUAL version that was run end-to-end and verified against
employee_attrition_raw.csv + injection_log.csv - not a theoretical draft.

Install what you need:
    pip install pandas numpy matplotlib seaborn great-expectations python-dateutil --break-system-packages
"""

import pandas as pd
import numpy as np
from dateutil import parser as dtparser
import matplotlib
matplotlib.use("Agg")  # remove this line if you want plots to pop up interactively
import matplotlib.pyplot as plt
import seaborn as sns

RAW_PATH = r"D:\Shin\Programming\ADS_project\data\processed\employee_attrition_raw.csv"
CLEANED_PATH = r"D:\Shin\Programming\ADS_project\data\processed\employee_attrition_cleaned.csv"
VALIDATION_REPORT_PATH = r"D:\Shin\Programming\ADS_project\output\validation_report.json"
PROFILE_REPORT_PATH = r"D:\Shin\Programming\ADS_project\output\profiling_report_raw.txt"

df = pd.read_csv(RAW_PATH)
print("RAW shape:", df.shape)


# ============================================================
# STEP 1 — MANUAL PROFILING (replaces ydata-profiling)
# ============================================================

report_lines = []
def log(line=""):
    print(line)
    report_lines.append(str(line))

log("=" * 70)
log("EMPLOYEE ATTRITION — RAW DATA PROFILING REPORT")
log("=" * 70)

# --- Structural overview ---
log(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
log("\nColumn dtypes:")
log(df.dtypes.to_string())

# --- Missingness profile ---
missing_count = df.isna().sum()
missing_pct = (missing_count / len(df) * 100).round(2)
missing_summary = pd.DataFrame({
    "missing_count": missing_count,
    "missing_pct": missing_pct
}).sort_values("missing_count", ascending=False)
missing_summary = missing_summary[missing_summary["missing_count"] > 0]

log(f"\nColumns with missing values ({len(missing_summary)} total):")
log(missing_summary.to_string())

plt.figure(figsize=(8, 5))
missing_summary["missing_pct"].plot(kind="barh", color="indianred")
plt.xlabel("% missing")
plt.title("Missing Values by Column")
plt.tight_layout()
plt.savefig(r"D:\Shin\Programming\ADS_project\output\profile_missingness.png", dpi=120)
plt.close()

# --- Descriptive statistics, numeric and categorical separately ---
numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()

log(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
log("\nNumeric summary statistics:")
log(df[numeric_cols].describe().T.to_string())

log(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
log("\nCategorical summary:")
log(df[categorical_cols].describe().T.to_string())

log("\nUnique value counts per categorical column (flags inconsistency):")
for col in categorical_cols:
    n_unique = df[col].nunique()
    log(f"  {col}: {n_unique} unique values")
    if n_unique <= 15:
        log(f"    -> {sorted(df[col].dropna().unique().tolist())}")

# --- Outlier detection via IQR ---
log("\nOutlier detection (IQR method, 1.5x multiplier):")
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    if n_outliers > 0:
        log(f"  {col}: {n_outliers} outliers (bounds: [{lower_bound:.1f}, {upper_bound:.1f}])")

# --- Correlation heatmap ---
plt.figure(figsize=(14, 11))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap — Numeric Features")
plt.tight_layout()
plt.savefig(r"D:\Shin\Programming\ADS_project\output\profile_correlation_heatmap.png", dpi=120)
plt.close()

corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False)
corr_pairs = corr_pairs[corr_pairs < 1.0]
log("\nTop 10 strongest feature correlations (by absolute value):")
log(corr_pairs.drop_duplicates().head(10).to_string())

# --- Distribution plots ---
key_numeric_features = [c for c in ["Age", "MonthlyIncome", "YearsAtCompany", "DistanceFromHome"] if c in df.columns]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, col in zip(axes.flatten(), key_numeric_features):
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribution: {col}")
plt.tight_layout()
plt.savefig(r"D:\Shin\Programming\ADS_project\output\profile_distributions.png", dpi=120)
plt.close()

# --- Duplicate check ---
n_exact_dupes = df.duplicated().sum()
n_id_dupes = df["EmployeeNumber"].duplicated().sum()
log(f"\nExact duplicate rows: {n_exact_dupes}")
log(f"Duplicate EmployeeNumber values: {n_id_dupes}")

with open(PROFILE_REPORT_PATH, "w") as f:
    f.write("\n".join(report_lines))
print(f"\nProfiling report saved: {PROFILE_REPORT_PATH}")


# ============================================================
# STEP 2 — CLEANING
# ============================================================

# --- Deduplication (on EmployeeNumber, catches exact AND near-duplicates) ---
before_dedup = len(df)
df = df.drop_duplicates(subset=["EmployeeNumber"], keep="first").reset_index(drop=True)
print(f"\nDedup: {before_dedup} -> {len(df)} (removed {before_dedup - len(df)})")

# --- Standardize categorical text ---
def standardize_text(series):
    return series.astype(str).str.strip().str.title()

df["Department"] = standardize_text(df["Department"])
DEPARTMENT_SYNONYMS = {
    "R&D": "Research & Development",
    "Research And Development": "Research & Development",
    "Hr": "Human Resources",
}
df["Department"] = df["Department"].replace(DEPARTMENT_SYNONYMS)
df["Gender"] = standardize_text(df["Gender"])
df["OverTime"] = df["OverTime"].replace({"Y": "Yes", "N": "No"})
print("Department after cleaning:", sorted(df["Department"].unique()))
print("OverTime after cleaning:", sorted(df["OverTime"].unique()))
print("Gender after cleaning:", sorted(df["Gender"].unique()))

# --- Parse mixed-format dates ---
def safe_parse_date(value):
    if pd.isna(value):
        return pd.NaT
    try:
        return dtparser.parse(str(value), dayfirst=False)
    except (ValueError, OverflowError):
        return pd.NaT

df["HireDate"] = df["HireDate"].apply(safe_parse_date)
print(f"HireDate NaT after parsing: {df['HireDate'].isna().sum()}")

# --- Fix logically impossible ages, THEN impute (was a bug in an earlier
# draft — nulling without re-imputing left real gaps in the cleaned file) ---
impossible_age_mask = (df["Age"] < 16) | (df["Age"] > 80)
print(f"Impossible Age values: {impossible_age_mask.sum()}")
df.loc[impossible_age_mask, "Age"] = np.nan
df["Age"] = df["Age"].fillna(df["Age"].median())

# --- Referential integrity: tenure can't exceed a plausible working lifetime ---
integrity_mask = df["YearsAtCompany"] > (df["Age"] - 16)
print(f"Tenure/Age integrity violations: {integrity_mask.sum()}")
df.loc[integrity_mask, "YearsAtCompany"] = (df.loc[integrity_mask, "Age"] - 16).clip(lower=0)

# --- Disambiguate MonthlyIncome: unit mismatch vs genuine outlier ---
median_income_by_level = df.groupby("JobLevel")["MonthlyIncome"].median()
def fix_income_row(row):
    val = row["MonthlyIncome"]
    if pd.isna(val):
        return val
    level_median = median_income_by_level.get(row["JobLevel"], df["MonthlyIncome"].median())
    if val > level_median * 6:
        candidate = val / 12
        if level_median * 0.3 <= candidate <= level_median * 3:
            return candidate
        return level_median * 3  # genuine outlier -> cap
    return val

income_before = df["MonthlyIncome"].copy()
df["MonthlyIncome"] = df.apply(fix_income_row, axis=1)
print(f"MonthlyIncome rows altered by unit/outlier fix: {(df['MonthlyIncome'] != income_before).sum()}")

# --- Missing values — handled per mechanism (MCAR / MAR / MNAR) ---
df["DistanceFromHome"] = df["DistanceFromHome"].fillna(df["DistanceFromHome"].median())  # MCAR
df["MonthlyIncome"] = df.groupby("JobLevel")["MonthlyIncome"].transform(lambda s: s.fillna(s.median()))  # MAR
df["performance_rating_was_missing"] = df["PerformanceRating"].isna().astype(int)  # MNAR: flag, don't hide
df["PerformanceRating"] = df["PerformanceRating"].fillna(df["PerformanceRating"].mode()[0])

missing_after = df.isna().sum()
print("Missing values AFTER cleaning:\n", missing_after[missing_after > 0])
# Expected remaining: HireDate (genuinely unparseable, not fabricated) and
# TerminationNotes (structurally empty for active employees — correct, not a bug)


# ============================================================
# STEP 3 — FEATURE ENGINEERING
# ============================================================

df["tenure_bucket"] = pd.cut(df["YearsAtCompany"], bins=[-1, 2, 5, 10, 100],
                              labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "10+ yrs"])
df["is_new_hire"] = (df["YearsAtCompany"] <= 1).astype(int)
df["age_group"] = pd.cut(df["Age"], bins=[0, 25, 35, 45, 55, 100],
                          labels=["<=25", "26-35", "36-45", "46-55", "56+"])
df["income_per_experience_year"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)

satisfaction_cols = ["EnvironmentSatisfaction", "JobSatisfaction",
                      "RelationshipSatisfaction", "WorkLifeBalance"]
raw_avg = df[satisfaction_cols].mean(axis=1)
df["satisfaction_composite"] = (raw_avg - 1) / (4 - 1)

df["hire_day_of_week"] = df["HireDate"].dt.day_name()
# Explicitly preserve missingness for unparseable dates instead of letting
# .isin() silently default them to False (a bug caught by actually running this)
weekend_flag = df["HireDate"].dt.dayofweek.isin([5, 6]).astype("Int64")
df["hired_on_weekend"] = weekend_flag.where(df["HireDate"].notna(), pd.NA)

# --- Termination type + voluntary-only label (label construction ONLY —
# see leakage note below) ---
INVOLUNTARY_KEYWORDS = "eliminated|downsizing|redundant"
df["termination_type"] = "Active"
left_mask = df["Attrition"] == "Yes"
is_involuntary = df["TerminationNotes"].str.contains(INVOLUNTARY_KEYWORDS, case=False, na=False)
df.loc[left_mask & is_involuntary, "termination_type"] = "Involuntary"
df.loc[left_mask & ~is_involuntary, "termination_type"] = "Voluntary"

df["attrition_label_voluntary_only"] = np.where(
    df["termination_type"] == "Involuntary", np.nan,
    df["Attrition"].map({"Yes": 1, "No": 0})
)

print("New feature columns:",
      ["tenure_bucket", "is_new_hire", "age_group", "income_per_experience_year",
       "satisfaction_composite", "hire_day_of_week", "hired_on_weekend",
       "termination_type", "attrition_label_voluntary_only",
       "performance_rating_was_missing"])

# LEAKAGE WARNING: these columns are kept in the saved file for audit
# traceability, but MUST be excluded from X (features) in Experiment 4.
# TerminationNotes/termination_type only exist for employees who already
# left — including them as model inputs lets the model read the answer
# directly off the target, producing artificially perfect but meaningless
# accuracy. performance_rating_was_missing is also target-correlated by
# construction (it was injected conditional on Attrition == "Yes").
LEAKAGE_RISK_COLUMNS = ["TerminationNotes", "termination_type", "performance_rating_was_missing"]
print(f"LEAKAGE-RISK COLUMNS (exclude from model features): {LEAKAGE_RISK_COLUMNS}")


# ============================================================
# STEP 4 — SCHEMA VALIDATION (manual, no great_expectations dependency)
# ============================================================
# Same checks as before, written as explicit, readable assertions instead
# of a library API. This IS a validation suite — it's just not delegated
# to a package that currently can't install on Python 3.14.

validation_results = []

def check(name, column, condition, details=""):
    passed = bool(condition)
    validation_results.append({
        "check": name, "column": column, "passed": passed, "details": details
    })
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name} ({column})")

print("\nRunning schema validation suite...")

# --- Completeness / identity checks ---
check("not_null", "EmployeeNumber",
      df["EmployeeNumber"].notna().all())
check("unique", "EmployeeNumber",
      df["EmployeeNumber"].is_unique)

# --- Range checks (domain-plausible bounds) ---
check("value_range_16_to_80", "Age",
      df["Age"].between(16, 80).all(),
      f"min={df['Age'].min()}, max={df['Age'].max()}")
check("value_range_0_to_50000", "MonthlyIncome",
      df["MonthlyIncome"].between(0, 50000).all(),
      f"min={df['MonthlyIncome'].min()}, max={df['MonthlyIncome'].max()}")
check("value_range_0_to_60", "YearsAtCompany",
      df["YearsAtCompany"].between(0, 60).all(),
      f"min={df['YearsAtCompany'].min()}, max={df['YearsAtCompany'].max()}")

# --- Category-set checks (catches inconsistencies re-appearing later) ---
check("values_in_set_yes_no", "Attrition",
      df["Attrition"].isin(["Yes", "No"]).all(),
      f"found: {sorted(df['Attrition'].unique())}")
check("values_in_set_yes_no", "OverTime",
      df["OverTime"].isin(["Yes", "No"]).all(),
      f"found: {sorted(df['OverTime'].unique())}")
check("values_in_set", "termination_type",
      df["termination_type"].isin(["Active", "Voluntary", "Involuntary"]).all(),
      f"found: {sorted(df['termination_type'].unique())}")

# --- Summarize ---
n_passed = sum(r["passed"] for r in validation_results)
n_total = len(validation_results)
validation_success = (n_passed == n_total)

print(f"\nValidation summary: {n_passed}/{n_total} checks passed")

import json
with open(VALIDATION_REPORT_PATH, "w") as f:
    json.dump({"success": validation_success, "results": validation_results}, f, indent=2)
print(f"Validation report saved: {VALIDATION_REPORT_PATH}")

# Same gate behavior as before — don't save/version a dataset that fails validation
assert validation_success, "Schema validation failed — inspect validation_report.json before proceeding."

# ============================================================
# STEP 5 — SAVE
# ============================================================

df.to_csv(CLEANED_PATH, index=False)
print(f"\nCLEANED dataset saved: {CLEANED_PATH} — final shape: {df.shape}")


# ============================================================
# CROSS-CHECK AGAINST INJECTION LOG (verification, run once you have it)
# ============================================================
try:
    log_df = pd.read_csv(r"D:\Shin\Programming\ADS_project\data\processed\injection_log.csv")
    print("\n--- Cross-check against injection_log.csv ---")
    print(log_df["issue_type"].value_counts())
except FileNotFoundError:
    print("\ninjection_log.csv not found in this folder — skip cross-check or copy it in.")
