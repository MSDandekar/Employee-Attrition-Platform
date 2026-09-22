"""
STEP 1 — MANUAL DATA PROFILING (replaces ydata-profiling)
============================================================
No auto-generated tool here — every check below is chosen deliberately to
answer a specific profiling question. This is the actual content a
profiling report needs to contain; we're just building it ourselves
instead of asking a library to guess what matters.

Produces: profiling_report_raw.txt (a plain-text summary) +
          3 PNG figures (missingness, correlation heatmap, distributions)
as your "Profiling report" deliverable.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RAW_PATH = "employee_attrition_raw.csv"
df = pd.read_csv(RAW_PATH)

report_lines = []
def log(line=""):
    print(line)
    report_lines.append(str(line))

log("=" * 70)
log("EMPLOYEE ATTRITION — RAW DATA PROFILING REPORT")
log("=" * 70)


# --- 1a. Structural overview: shape, dtypes, memory ---
# WHY THIS FIRST: before touching values, you need to know what you're
# looking at — how many rows/columns, what type pandas inferred for each
# (a column read as 'object' when it should be numeric is itself a data
# quality signal, not just metadata).
log(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
log("\nColumn dtypes:")
log(df.dtypes.to_string())


# --- 1b. Missingness profile ---
# .isna().sum() gives counts; we also compute PERCENTAGES, because "12
# missing values" means something different in a 40-row dataset than a
# 4000-row one. Sorting descending surfaces the worst offenders first.
missing_count = df.isna().sum()
missing_pct = (missing_count / len(df) * 100).round(2)
missing_summary = pd.DataFrame({
    "missing_count": missing_count,
    "missing_pct": missing_pct
}).sort_values("missing_count", ascending=False)
missing_summary = missing_summary[missing_summary["missing_count"] > 0]

log(f"\nColumns with missing values ({len(missing_summary)} total):")
log(missing_summary.to_string())

# Visual: a missingness bar chart makes gaps obvious at a glance, and is
# the standard first figure in any real profiling report.
plt.figure(figsize=(8, 5))
missing_summary["missing_pct"].plot(kind="barh", color="indianred")
plt.xlabel("% missing")
plt.title("Missing Values by Column")
plt.tight_layout()
plt.savefig("profile_missingness.png", dpi=120)
plt.close()


# --- 1c. Descriptive statistics: numeric and categorical, SEPARATELY ---
# .describe() defaults to numeric columns only. We explicitly ALSO run it
# on categoricals (include='object') because count/unique/top/freq answers
# a completely different question — "are there suspiciously many unique
# categories, or one category dominating unexpectedly?"
numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()

log(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
log("\nNumeric summary statistics:")
log(df[numeric_cols].describe().T.to_string())

log(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
log("\nCategorical summary (count / unique / most frequent / its frequency):")
log(df[categorical_cols].describe().T.to_string())

# A HIGH unique-count on a column that should have few categories (e.g.
# Department showing 12 unique values instead of 3) is exactly the kind
# of category-inconsistency signal we deliberately injected — this is
# where you'd catch it, in your OWN report, before writing cleaning code.
log("\nUnique value counts per categorical column (flags inconsistency):")
for col in categorical_cols:
    n_unique = df[col].nunique()
    log(f"  {col}: {n_unique} unique values")
    if n_unique <= 15:  # only print the actual values if the list is short enough to read
        log(f"    -> {sorted(df[col].dropna().unique().tolist())}")


# --- 1d. Outlier detection via IQR (Interquartile Range) method ---
# WHY IQR OVER JUST EYEBALLING describe(): the IQR method gives you an
# OBJECTIVE, REPRODUCIBLE rule (any value beyond 1.5x the interquartile
# range from Q1/Q3 is flagged) instead of a subjective "that number looks
# big." This is the standard statistical definition of an outlier, and
# it's what you cite when someone asks "how did you decide that was an
# outlier?"
log("\nOutlier detection (IQR method, 1.5x multiplier):")
outlier_summary = {}
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    if n_outliers > 0:
        outlier_summary[col] = n_outliers
        log(f"  {col}: {n_outliers} outliers (bounds: [{lower_bound:.1f}, {upper_bound:.1f}])")


# --- 1e. Correlation heatmap (numeric features only) ---
# Purpose: spot multicollinearity BEFORE modeling (two features that are
# ~1.0 correlated are redundant and can destabilize some models), and
# sanity-check that relationships look plausible (e.g. Age and
# TotalWorkingYears SHOULD correlate positively — if they didn't, that
# would itself be a red flag worth investigating).
plt.figure(figsize=(14, 11))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap — Numeric Features")
plt.tight_layout()
plt.savefig("profile_correlation_heatmap.png", dpi=120)
plt.close()

# Explicitly call out the strongest pairs — a heatmap alone makes you
# scan visually; this makes the top relationships a written finding.
corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False)
corr_pairs = corr_pairs[corr_pairs < 1.0]  # drop self-correlation (always 1.0)
log("\nTop 10 strongest feature correlations (by absolute value):")
log(corr_pairs.drop_duplicates().head(10).to_string())


# --- 1f. Distribution plots for key numeric features ---
# A histogram shows shape (skew, multi-modality) that a single mean/median
# number in describe() hides. We only plot the columns most relevant to
# attrition analysis, not all 26, to keep the report focused and readable.
key_numeric_features = ["Age", "MonthlyIncome", "YearsAtCompany", "DistanceFromHome"]
key_numeric_features = [c for c in key_numeric_features if c in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, col in zip(axes.flatten(), key_numeric_features):
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribution: {col}")
plt.tight_layout()
plt.savefig("profile_distributions.png", dpi=120)
plt.close()


# --- 1g. Duplicate check ---
# Two DIFFERENT duplicate questions, deliberately kept separate: exact
# full-row duplicates (rare, obvious) vs. duplicate IDENTITY (same
# EmployeeNumber appearing more than once — the more dangerous kind,
# since the rows aren't identical so drop_duplicates() alone won't catch
# it without specifying the subset).
n_exact_dupes = df.duplicated().sum()
n_id_dupes = df["EmployeeNumber"].duplicated().sum()
log(f"\nExact duplicate rows (all columns identical): {n_exact_dupes}")
log(f"Duplicate EmployeeNumber values (same identity, possibly different data): {n_id_dupes}")


# --- Save the full text report ---
with open("profiling_report_raw.txt", "w") as f:
    f.write("\n".join(report_lines))

print("\n" + "=" * 70)
print("Profiling artifacts saved:")
print("  - profiling_report_raw.txt")
print("  - profile_missingness.png")
print("  - profile_correlation_heatmap.png")
print("  - profile_distributions.png")
