"""
exp5_XAI_fairness.py
======================
Experiment 5 — Explainable AI (SHAP) + Fairness Audit (Fairlearn)
Built on the leakage-fixed Experiment 4 dataset (attrition_label_voluntary_only
as target). Verified end-to-end against the real cleaned dataset (1470 rows,
50 cols; target split 1233 stayed / 185 voluntary-left / 52 involuntary-set-aside).

CHANGE LOG (this revision)
---------------------------
- LIME REMOVED ENTIRELY. Rationale: LIME approximates a model with a local
  linear surrogate. The primary model (Logistic Regression) IS already
  linear/additive, so SHAP's LinearExplainer gives the EXACT decomposition
  of every prediction with zero approximation error. Using LIME on top of
  an already-additive model adds approximation error to explain something
  that's already exactly explainable — strictly worse, not just broken.
  (It was also breaking with a scipy truncnorm domain error in its internal
  discretizer, which is a separate, known LIME fragility issue — not fixed,
  because fixing it wouldn't have made it a better choice than what follows.)
- NEW: explain_employee_risk() — a per-employee, readable, signed feature
  contribution breakdown (same style as the pairwise "ranking difference"
  function you liked), now run for ALL 8 curated employees individually,
  not just pairwise. Produces both a CSV (long format, for the report
  appendix) and a plain-text narrative block (drop straight into your
  writeup) and a per-employee tornado bar chart PNG.
- SHAP visuals redesigned: bigger, centered, single plot per figure,
  diverging color by direction of effect, larger fonts, and a bug fix
  (the old plot_shap_summary computed a `colors` list and never used it —
  every bar was rendered the same flat blue regardless of sign).
- IMPORTANT UNIT NOTE: all SHAP contribution values below are in LOG-ODDS
  (logit) units, not probability. risk_score (0-1) IS probability. These
  are two different, correctly-related but not directly comparable scales
  — this is now labeled explicitly everywhere it's printed/plotted, since
  the original script left this ambiguous.
- IMPORTANT BASELINE NOTE: the per-employee SHAP decomposition is anchored
  to explainer.expected_value, which represents the average prediction over
  the TRAINING background sample (this is required for the shap values to
  sum exactly to the employee's raw score — that's the whole point of using
  SHAP here). Separately, we also report each employee's risk_score against
  the mean predicted risk on the TEST set, since that's the benchmark you
  asked for. These are two different reference points answering two
  different questions — don't conflate them in the write-up.

Models explained: Logistic Regression (PRIMARY — exact, additive risk
decomposition, enables ranking explanations) and Random Forest (secondary
challenger, richer non-linear SHAP dependence plots).

Sensitive attributes: Gender, MaritalStatus, age_group (all REAL IBM fields).
race_ethnicity/manager_id are EXCLUDED ENTIRELY — synthetic, sampled
independently of everything else in Experiment 2; any fairness finding on
them would be analytically meaningless.

Install what you need:
    pip install shap fairlearn --break-system-packages
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from fairlearn.metrics import (MetricFrame, demographic_parity_difference,
                                 equalized_odds_difference, selection_rate)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
DATA_PATH = r"D:\Shin\Programming\ADS_project\data\processed\employee_attrition_cleaned.csv"
OUTPUT_DIR = r"D:\Shin\Programming\ADS_project\output.1"
EMPLOYEE_PLOT_DIR = os.path.join(OUTPUT_DIR, "employee_explanations")
os.makedirs(EMPLOYEE_PLOT_DIR, exist_ok=True)

# A small, consistent visual language used across every plot in this file,
# so a professor/HR reader sees the same color = same meaning everywhere.
COLOR_INCREASES_RISK = "#C0392B"   # warm red  -> pushes risk UP
COLOR_DECREASES_RISK = "#2E5A88"   # cool blue -> pushes risk DOWN
FONT_TITLE = 16
FONT_LABEL = 13
FONT_TICK = 11


# ============================================================
# STEP 1 — DATASET & MODEL SELECTION
# ============================================================
df = pd.read_csv(DATA_PATH)
layoffs_df = df[df["termination_type"] == "Involuntary"].copy()
df = df[df["attrition_label_voluntary_only"].notna()].copy().reset_index(drop=True)
print(f"Modeling rows: {len(df)} | Layoffs set aside: {len(layoffs_df)}")

LEAKAGE_RISK_COLUMNS = ["TerminationNotes", "termination_type", "performance_rating_was_missing", "Attrition"]
IDENTITY_COLUMNS = ["EmployeeNumber", "HireDate", "hire_day_of_week", "Over18",
                     "EmployeeCount", "StandardHours", "manager_id", "race_ethnicity",
                     "attrition_label_voluntary_only"]  # target itself — must stay excluded from X
DROP_COLUMNS = [c for c in (LEAKAGE_RISK_COLUMNS + IDENTITY_COLUMNS) if c in df.columns]

X = df.drop(columns=DROP_COLUMNS)
y = df["attrition_label_voluntary_only"].astype(int)
print("Feature matrix X:", X.shape)

categorical_features = X.select_dtypes(include="object").columns.tolist()
numeric_features = X.select_dtypes(include="number").columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
y_test_reset = y_test.reset_index(drop=True)
X_test_reset = X_test.reset_index(drop=True)

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# PRIMARY model — Logistic Regression, best hyperparameters from Experiment 4
logreg_pipe = Pipeline([("preprocess", preprocessor),
                         ("model", LogisticRegression(class_weight="balanced", max_iter=1000,
                                                       random_state=RANDOM_SEED, C=0.0746, solver="lbfgs"))])
# SECONDARY challenger — Random Forest, best hyperparameters from Experiment 4
rf_pipe = Pipeline([("preprocess", preprocessor),
                     ("model", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED,
                                                       n_jobs=1, n_estimators=370, max_depth=9, min_samples_leaf=4))])

logreg_pipe.fit(X_train, y_train)
rf_pipe.fit(X_train, y_train)

for name, pipe in [("LogisticRegression (PRIMARY)", logreg_pipe), ("RandomForest (challenger)", rf_pipe)]:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    print(f"{name}: accuracy={accuracy_score(y_test, pred):.3f}, roc_auc={roc_auc_score(y_test, proba):.3f}")


# ============================================================
# STEP 2 — EXPLAINABILITY WITH SHAP
# ============================================================
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
if hasattr(X_train_transformed, "toarray"):
    X_train_transformed = X_train_transformed.toarray()
    X_test_transformed = X_test_transformed.toarray()
feat_names = preprocessor.get_feature_names_out()

# LinearExplainer for Logistic Regression — EXACT, not approximate, for
# linear models. shap_values here are in LOG-ODDS units (the model's raw
# linear score space), and sum exactly to (raw_score - expected_value).
explainer_lr = shap.LinearExplainer(logreg_pipe.named_steps["model"], X_train_transformed)
shap_values_lr = explainer_lr.shap_values(X_test_transformed)
baseline_logodds_lr = explainer_lr.expected_value
baseline_prob_lr = 1 / (1 + np.exp(-baseline_logodds_lr))  # for reference only

# TreeExplainer for Random Forest — exact and fast for tree ensembles
explainer_rf = shap.TreeExplainer(rf_pipe.named_steps["model"])
shap_values_rf_raw = explainer_rf.shap_values(X_test_transformed)
shap_values_rf = shap_values_rf_raw[:, :, 1]  # class 1 = "will leave"

# --- Aggregate one-hot dummy SHAP values back to their ORIGINAL columns ---
# WHY: without this, a SHAP plot shows "Department_Sales", "Department_HR",
# "Department_R&D" as three separate bars — confusing for an HR reader and
# not how the business thinks about the feature. We sum each dummy's SHAP
# contribution back onto its single source column.
import re
def get_original_column(transformed_name):
    name = re.sub(r"^(num__|cat__)", "", transformed_name)
    if transformed_name.startswith("cat__"):
        for orig_col in categorical_features:
            if name.startswith(orig_col + "_"):
                return orig_col
    return name

mapping = {fn: get_original_column(fn) for fn in feat_names}

def aggregate_shap_to_original(shap_vals, feat_names, mapping):
    df_shap = pd.DataFrame(shap_vals, columns=feat_names)
    agg = pd.DataFrame(index=df_shap.index)
    for orig_col in dict.fromkeys(mapping.values()):
        cols = [fn for fn, oc in mapping.items() if oc == orig_col]
        agg[orig_col] = df_shap[cols].sum(axis=1)
    return agg

agg_shap_lr = aggregate_shap_to_original(shap_values_lr, feat_names, mapping)
agg_shap_rf = aggregate_shap_to_original(shap_values_rf, feat_names, mapping)


# --- Deliverable: SHAP summary plots — redesigned ---
# Bigger, single centered figure. Bars colored by the SIGN of the average
# effect (red = tends to push risk up, blue = tends to push risk down),
# which the previous version computed but never actually applied.
def plot_shap_summary(agg_shap_df, title, filename, top_n=15):
    mean_signed = agg_shap_df.mean().sort_values(key=abs, ascending=False).head(top_n)
    mean_signed = mean_signed.iloc[::-1]  # largest at top when plotted horizontally
    colors = [COLOR_INCREASES_RISK if v > 0 else COLOR_DECREASES_RISK for v in mean_signed.values]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    bars = ax.barh(mean_signed.index, mean_signed.values, color=colors, edgecolor="white", height=0.7)

    # Value labels at the end of each bar so exact numbers don't require
    # eyeballing the axis.
    for bar, val in zip(bars, mean_signed.values):
        offset = 0.01 * (mean_signed.abs().max())
        x = bar.get_width() + (offset if val >= 0 else -offset)
        ha = "left" if val >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}",
                 va="center", ha=ha, fontsize=FONT_TICK)

    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("Mean SHAP contribution (log-odds units) — average effect on risk score",
                  fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=14)
    ax.tick_params(axis="y", labelsize=FONT_LABEL)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Legend explaining the color language, since color-only encodings need
    # a key for anyone seeing this for the first time.
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_INCREASES_RISK),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_DECREASES_RISK)]
    ax.legend(handles, ["Increases predicted attrition risk", "Decreases predicted attrition risk"],
              loc="lower right", fontsize=FONT_TICK, frameon=True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)

plot_shap_summary(agg_shap_lr, "SHAP Global Feature Importance — Logistic Regression (Primary Model)",
                   "shap_summary_logreg.png")
plot_shap_summary(agg_shap_rf, "SHAP Global Feature Importance — Random Forest (Challenger Model)",
                   "shap_summary_rf.png")


# --- Deliverable: SHAP dependence plot for the top feature — redesigned ---
# Style branches depending on whether the top feature is categorical or
# numeric, because a scatter plot of a categorical feature (the old
# behaviour) just produces an unreadable vertical smear of points.
top_feature_lr = agg_shap_lr.abs().mean().idxmax()
is_categorical_top = top_feature_lr in categorical_features

fig, ax = plt.subplots(figsize=(10, 6.5))
if is_categorical_top:
    categories = X_test_reset[top_feature_lr].astype(str)
    order = categories.value_counts().index.tolist()
    data_by_cat = [agg_shap_lr[top_feature_lr][categories == cat] for cat in order]
    bp = ax.boxplot(data_by_cat, labels=order, patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("#7FA6C9")
        patch.set_alpha(0.8)
    for median in bp["medians"]:
        median.set_color("#1A1A1A")
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel(top_feature_lr, fontsize=FONT_LABEL)
else:
    x_vals = X_test_reset[top_feature_lr]
    y_vals = agg_shap_lr[top_feature_lr]
    colors = [COLOR_INCREASES_RISK if v > 0 else COLOR_DECREASES_RISK for v in y_vals]
    ax.scatter(x_vals, y_vals, alpha=0.55, s=45, c=colors, edgecolor="none")
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel(top_feature_lr, fontsize=FONT_LABEL)

ax.set_ylabel("SHAP contribution to risk (log-odds)", fontsize=FONT_LABEL)
ax.set_title(f"How {top_feature_lr} Affects Predicted Attrition Risk\n(Logistic Regression, Primary Model)",
             fontsize=FONT_TITLE, fontweight="bold")
ax.tick_params(axis="both", labelsize=FONT_TICK)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, f"shap_dependence_{top_feature_lr}.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nTop feature by mean |SHAP| (LogReg): {top_feature_lr}")
print("Top 10 features (LogReg):\n", agg_shap_lr.abs().mean().sort_values(ascending=False).head(10))
print("\nTop 10 features (RandomForest):\n", agg_shap_rf.abs().mean().sort_values(ascending=False).head(10))


# ============================================================
# PER-EMPLOYEE EXPLANATIONS — the replacement for LIME
# ============================================================
# Because risk_score = intercept + sum(coef_i * feature_i), each employee's
# score can be decomposed EXACTLY, feature by feature, against a fixed
# reference point (explainer.expected_value — the average prediction over
# the training background sample). This is what makes a genuinely exact,
# human-readable per-employee explanation possible with this model, with
# zero approximation error — the property LIME cannot match.

def explain_employee_risk(idx, agg_shap_df, results_df, mean_test_risk, top_n=8):
    """
    Returns (narrative_lines, records) for one employee.
    narrative_lines: list[str] ready to print or write to a text file,
        formatted the same way as the pairwise version you liked:
        "FeatureName: +0.919 (higher risk contribution)"
    records: list[dict] ready to append to a long-format CSV.
    """
    row = results_df.loc[idx]
    contributions = agg_shap_df.loc[idx].sort_values(key=abs, ascending=False)

    header = (
        f"Employee row {idx} — predicted risk = {row['risk_score']:.3f} "
        f"(actual: {'Left' if row['actual'] == 1 else 'Stayed'}, "
        f"predicted: {'Left' if row['predicted'] == 1 else 'Stayed'})"
    )
    vs_mean = row['risk_score'] - mean_test_risk
    comparison = (
        f"  This employee's predicted risk is {abs(vs_mean):.3f} "
        f"{'above' if vs_mean >= 0 else 'below'} the mean predicted risk "
        f"across the test set ({mean_test_risk:.3f})."
    )
    note = "  (Feature contributions below are in log-odds units, relative to the model's average baseline prediction — not a direct % scale.)"

    lines = [header, comparison, note, "  Top contributing features:"]
    records = []
    for rank, (feat, val) in enumerate(contributions.head(top_n).items(), start=1):
        direction = "higher risk contribution" if val > 0 else "lower risk contribution"
        lines.append(f"    {rank}. {feat}: {val:+.3f} ({direction})")
        records.append({
            "employee_row": idx, "risk_score": row["risk_score"],
            "actual": row["actual"], "predicted": row["predicted"],
            "feature_rank": rank, "feature": feat, "shap_value_logodds": val,
            "direction": direction,
        })
    return lines, records


def plot_employee_tornado(idx, agg_shap_df, results_df, filename, top_n=8):
    """One clean, centered horizontal 'tornado' bar chart per employee —
    the direct visual replacement for LIME's per-instance bar chart."""
    row = results_df.loc[idx]
    contributions = agg_shap_df.loc[idx].sort_values(key=abs, ascending=False).head(top_n)
    contributions = contributions.iloc[::-1]
    colors = [COLOR_INCREASES_RISK if v > 0 else COLOR_DECREASES_RISK for v in contributions.values]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(contributions.index, contributions.values, color=colors, edgecolor="white", height=0.65)
    for bar, val in zip(bars, contributions.values):
        offset = 0.02 * max(contributions.abs().max(), 0.01)
        x = bar.get_width() + (offset if val >= 0 else -offset)
        ha = "left" if val >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}", va="center", ha=ha, fontsize=FONT_TICK)

    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("SHAP contribution (log-odds units)", fontsize=FONT_LABEL)
    ax.set_title(f"Why Employee (row {idx}) Has a Predicted Risk of {row['risk_score']:.1%}",
                 fontsize=FONT_TITLE, fontweight="bold", pad=14)
    ax.tick_params(axis="both", labelsize=FONT_LABEL)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_INCREASES_RISK),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_DECREASES_RISK)]
    ax.legend(handles, ["Pushes risk up", "Pushes risk down"], loc="lower right",
              fontsize=FONT_TICK, frameon=True)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- Curated selection: 5 highest-ranked-risk employees + 3 wrong predictions ---
proba_test = logreg_pipe.predict_proba(X_test)[:, 1]
y_pred_lr = logreg_pipe.predict(X_test)
results_df = X_test.copy().reset_index(drop=True)
results_df["actual"] = y_test_reset.values
results_df["predicted"] = y_pred_lr
results_df["risk_score"] = proba_test

mean_test_risk = float(proba_test.mean())

top5_idx = results_df.sort_values("risk_score", ascending=False).head(5).index.tolist()
wrong_mask = results_df["actual"] != results_df["predicted"]
wrong_idx = results_df[wrong_mask].sample(n=min(3, wrong_mask.sum()), random_state=RANDOM_SEED).index.tolist()
curated_idx = list(dict.fromkeys(top5_idx + wrong_idx))
print(f"\nCurated explanation set ({len(curated_idx)} employees):")
print(results_df.loc[curated_idx, ["actual", "predicted", "risk_score"]])
print(f"Mean predicted risk across full test set: {mean_test_risk:.3f}")

# --- Generate the explanation for all 8 curated employees ---
all_lines = [
    "PER-EMPLOYEE RISK EXPLANATIONS — Logistic Regression (Primary Model)",
    f"Mean predicted risk across test set (n={len(results_df)}): {mean_test_risk:.3f}",
    "=" * 70,
]
all_records = []
for idx in curated_idx:
    tag = "wrong_pred" if idx in wrong_idx else "top_risk"
    lines, records = explain_employee_risk(idx, agg_shap_lr, results_df, mean_test_risk, top_n=8)
    all_lines.extend(lines)
    all_lines.append("")  # blank line between employees
    for r in records:
        r["tag"] = tag
    all_records.extend(records)

    plot_path = os.path.join(EMPLOYEE_PLOT_DIR, f"employee_{tag}_row{idx}.png")
    plot_employee_tornado(idx, agg_shap_lr, results_df, plot_path, top_n=8)

# Save narrative text (drop straight into the report / appendix)
narrative_path = os.path.join(OUTPUT_DIR, "employee_risk_explanations.txt")
with open(narrative_path, "w", encoding="utf-8") as f:
    f.write("\n".join(all_lines))

# Save long-format CSV (feature-level rows, for tables/analysis)
explanations_df = pd.DataFrame(all_records)
explanations_df.to_csv(os.path.join(OUTPUT_DIR, "employee_risk_explanations.csv"), index=False)

print(f"\nSaved {len(curated_idx)} employee explanations:")
print(f"  - {narrative_path}")
print(f"  - {os.path.join(OUTPUT_DIR, 'employee_risk_explanations.csv')}")
print(f"  - {EMPLOYEE_PLOT_DIR}\\employee_*_row*.png ({len(curated_idx)} charts)")


# ============================================================
# STEP 4 — FAIRNESS AUDIT WITH FAIRLEARN
# ============================================================
# Sensitive attributes: REAL fields only (Gender, MaritalStatus, age_group).
# race_ethnicity/manager_id are deliberately excluded — see module docstring.
sensitive_attributes = {
    "Gender": X_test["Gender"].reset_index(drop=True),
    "MaritalStatus": X_test["MaritalStatus"].reset_index(drop=True),
    "age_group": X_test["age_group"].reset_index(drop=True),
}

fairness_report_rows = []
for attr_name, sensitive in sensitive_attributes.items():
    dpd = demographic_parity_difference(y_test_reset, y_pred_lr, sensitive_features=sensitive)
    eod = equalized_odds_difference(y_test_reset, y_pred_lr, sensitive_features=sensitive)
    mf = MetricFrame(metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
                      y_true=y_test_reset, y_pred=y_pred_lr, sensitive_features=sensitive)
    subgroup_sizes = sensitive.value_counts()

    print(f"\n=== Fairness audit: {attr_name} ===")
    print(f"Demographic parity difference: {dpd:.4f}")
    print(f"Equalized odds difference: {eod:.4f}")
    print(mf.by_group)
    print("Subgroup sizes (flag any < 20 as low-confidence):\n", subgroup_sizes)

    for group in mf.by_group.index:
        fairness_report_rows.append({
            "sensitive_attribute": attr_name, "group": group,
            "n_in_group": subgroup_sizes.get(group, np.nan),
            "low_confidence": subgroup_sizes.get(group, 0) < 20,
            "accuracy": mf.by_group.loc[group, "accuracy"],
            "selection_rate": mf.by_group.loc[group, "selection_rate"],
            "demographic_parity_difference": dpd,
            "equalized_odds_difference": eod,
        })

fairness_report_df = pd.DataFrame(fairness_report_rows)
fairness_report_df.to_csv(os.path.join(OUTPUT_DIR, "fairness_audit_report.csv"), index=False)
print(f"\nFairness audit report saved: {os.path.join(OUTPUT_DIR, 'fairness_audit_report.csv')}")

# --- Deliverable: fairness visualization ---
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
for ax, (attr_name, sensitive) in zip(axes, sensitive_attributes.items()):
    mf = MetricFrame(metrics={"selection_rate": selection_rate},
                      y_true=y_test_reset, y_pred=y_pred_lr, sensitive_features=sensitive)
    mf.by_group["selection_rate"].plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_title(f"Selection Rate by {attr_name}", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Selection rate (flagged as at-risk)", fontsize=FONT_TICK)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30, labelsize=FONT_TICK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
fig.suptitle("Fairness Audit — Selection Rate by Protected Attribute", fontsize=FONT_TITLE, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fairness_selection_rates.png"), dpi=150, bbox_inches="tight")
plt.close(fig)


# ============================================================
# STEP 5 — BIAS MITIGATION
# ============================================================
# MaritalStatus showed the largest, most reliably-sized disparity
# (Demographic parity difference ≈ 0.36 — Single employees flagged at
# nearly 4x the rate of Divorced employees) — this is the one we mitigate
# concretely, not just propose in writing.
print("\n=== BIAS MITIGATION: Post-processing via Fairlearn's ThresholdOptimizer ===")
print("Target attribute: MaritalStatus (largest, most reliable disparity found)")

mitigator = ThresholdOptimizer(
    estimator=logreg_pipe, constraints="demographic_parity",
    predict_method="predict_proba", prefit=True
)
mitigator.fit(X_train, y_train, sensitive_features=X_train["MaritalStatus"])
y_pred_mitigated = mitigator.predict(X_test, sensitive_features=X_test["MaritalStatus"])

sensitive_marital = X_test["MaritalStatus"].reset_index(drop=True)
dpd_before = demographic_parity_difference(y_test_reset, y_pred_lr, sensitive_features=sensitive_marital)
dpd_after = demographic_parity_difference(y_test_reset, pd.Series(y_pred_mitigated).reset_index(drop=True),
                                            sensitive_features=sensitive_marital)
acc_before = accuracy_score(y_test_reset, y_pred_lr)
acc_after = accuracy_score(y_test_reset, y_pred_mitigated)

mitigation_summary = pd.DataFrame({
    "metric": ["demographic_parity_difference", "accuracy"],
    "before_mitigation": [dpd_before, acc_before],
    "after_mitigation": [dpd_after, acc_after],
})
mitigation_summary.to_csv(os.path.join(OUTPUT_DIR, "mitigation_comparison.csv"), index=False)
print(mitigation_summary)

# --- Other mitigation strategies to PROPOSE in your report, not implemented here ---
print("""
Additional mitigation strategies to discuss in your report (not implemented,
per the rubric's "propose if detected" wording — one was implemented above
as a concrete demonstration; these remain proposals):
  - Pre-processing: reweight training samples so Single/Married/Divorced
    employees contribute equally to the loss function, rather than adjusting
    predictions after training.
  - In-processing: Fairlearn's ExponentiatedGradient with a demographic
    parity constraint, applied DURING training rather than after.
  - Post-processing (alternative): per-group threshold tuning targeting
    equalized odds instead of demographic parity, if false-negative rate
    parity matters more than flagging-rate parity for this use case.
""")

print("\n=== EXPERIMENT 5 COMPLETE ===")
print("Deliverables saved:")
print(f"  - {OUTPUT_DIR}\\shap_summary_logreg.png, shap_summary_rf.png")
print(f"  - {OUTPUT_DIR}\\shap_dependence_{top_feature_lr}.png")
print(f"  - {OUTPUT_DIR}\\employee_risk_explanations.txt, employee_risk_explanations.csv")
print(f"  - {EMPLOYEE_PLOT_DIR}\\employee_*_row*.png ({len(curated_idx)} per-employee charts)")
print(f"  - {OUTPUT_DIR}\\fairness_audit_report.csv, fairness_selection_rates.png")
print(f"  - {OUTPUT_DIR}\\mitigation_comparison.csv")
