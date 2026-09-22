"""
exp5_XAI_fairness.py
======================
Experiment 5 — Explainable AI (SHAP & LIME) + Fairness Audit (Fairlearn)
Built on the leakage-fixed Experiment 4 dataset (attrition_label_voluntary_only
as target). Verified end-to-end against your real cleaned dataset.

Models explained: Logistic Regression (PRIMARY — exact, additive risk
decomposition, enables ranking explanations) and Random Forest (secondary
challenger, richer non-linear SHAP dependence plots).

Sensitive attributes: Gender, MaritalStatus, age_group (all REAL IBM fields).
race_ethnicity/manager_id are EXCLUDED ENTIRELY — they were synthetic
(sampled independently of everything else in Experiment 2) and any fairness
finding on them would be analytically meaningless.

Install what you need:
    pip install shap lime fairlearn --break-system-packages
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
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
OUTPUT_DIR = r"D:\Shin\Programming\ADS_project\output"


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

# LinearExplainer for Logistic Regression — exact, not approximate, for linear models
explainer_lr = shap.LinearExplainer(logreg_pipe.named_steps["model"], X_train_transformed)
shap_values_lr = explainer_lr.shap_values(X_test_transformed)

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

# --- Deliverable: SHAP summary plots (aggregated, original-column names) ---
def plot_shap_summary(agg_shap_df, X_original, title, filename):
    mean_abs = agg_shap_df.abs().mean().sort_values(ascending=False)
    top15 = mean_abs.head(15)
    plt.figure(figsize=(9, 7))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in
              agg_shap_df[top15.index].mean()]  # rough direction indicator
    plt.barh(top15.index[::-1], top15.values[::-1], color="#4C72B0")
    plt.xlabel("Mean |SHAP value| (average impact on risk score)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}\\{filename}", dpi=130)
    plt.close()

plot_shap_summary(agg_shap_lr, X_test, "SHAP Global Feature Importance — Logistic Regression (Primary)",
                   "shap_summary_logreg.png")
plot_shap_summary(agg_shap_rf, X_test, "SHAP Global Feature Importance — Random Forest (Challenger)",
                   "shap_summary_rf.png")

# --- Deliverable: SHAP dependence plot for the top feature ---
top_feature_lr = agg_shap_lr.abs().mean().idxmax()
plt.figure(figsize=(8, 5))
plt.scatter(X_test[top_feature_lr].astype(str) if X_test[top_feature_lr].dtype == object
            else X_test[top_feature_lr], agg_shap_lr[top_feature_lr], alpha=0.5, s=20)
plt.xlabel(top_feature_lr)
plt.ylabel(f"SHAP value for {top_feature_lr}")
plt.title(f"SHAP Dependence Plot — {top_feature_lr} (Logistic Regression)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}\\shap_dependence_{top_feature_lr}.png", dpi=130)
plt.close()

print(f"\nTop feature by mean |SHAP| (LogReg): {top_feature_lr}")
print("Top 10 features (LogReg):\n", agg_shap_lr.abs().mean().sort_values(ascending=False).head(10))
print("\nTop 10 features (RandomForest):\n", agg_shap_rf.abs().mean().sort_values(ascending=False).head(10))


# ============================================================
# RANKING EXPLANATION — the payoff of choosing Logistic Regression
# ============================================================
# Because risk_score = intercept + sum(coef_i * feature_i), any two
# employees' score difference can be decomposed EXACTLY, feature by
# feature — this is what makes explaining a RANKING (not just a single
# prediction) genuinely feasible with this model.
def explain_ranking_difference(idx_a, idx_b, agg_shap_df, results_df, top_n=5):
    diff = (agg_shap_df.loc[idx_a] - agg_shap_df.loc[idx_b]).sort_values(key=abs, ascending=False)
    print(f"\nWhy employee {idx_a} (risk={results_df.loc[idx_a,'risk_score']:.3f}) ranks "
          f"{'above' if results_df.loc[idx_a,'risk_score'] > results_df.loc[idx_b,'risk_score'] else 'below'} "
          f"employee {idx_b} (risk={results_df.loc[idx_b,'risk_score']:.3f}):")
    for feat, val in diff.head(top_n).items():
        direction = "higher risk contribution" if val > 0 else "lower risk contribution"
        print(f"  {feat}: {val:+.3f} ({direction} for employee {idx_a})")


# ============================================================
# STEP 3 — EXPLAINABILITY WITH LIME
# ============================================================
cat_feature_indices = [i for i, fn in enumerate(feat_names) if fn.startswith("cat__")]
explainer_lime = LimeTabularExplainer(
    training_data=X_train_transformed, feature_names=list(feat_names),
    class_names=["Stayed", "Left"], categorical_features=cat_feature_indices,
    mode="classification", random_state=RANDOM_SEED,
)

# --- Curated selection: 5 highest-ranked-risk employees + 3 wrong predictions ---
proba_test = logreg_pipe.predict_proba(X_test)[:, 1]
y_pred_lr = logreg_pipe.predict(X_test)
results_df = X_test.copy().reset_index(drop=True)
results_df["actual"] = y_test_reset.values
results_df["predicted"] = y_pred_lr
results_df["risk_score"] = proba_test

top5_idx = results_df.sort_values("risk_score", ascending=False).head(5).index.tolist()
wrong_mask = results_df["actual"] != results_df["predicted"]
wrong_idx = results_df[wrong_mask].sample(n=min(3, wrong_mask.sum()), random_state=RANDOM_SEED).index.tolist()
curated_idx = list(dict.fromkeys(top5_idx + wrong_idx))
print(f"\nCurated LIME/ranking examples ({len(curated_idx)} employees):")
print(results_df.loc[curated_idx, ["actual", "predicted", "risk_score"]])

# --- Deliverable: LIME local explanation visualizations, saved as HTML ---
for i, idx in enumerate(curated_idx):
    exp = explainer_lime.explain_instance(
        X_test_transformed[idx], logreg_pipe.named_steps["model"].predict_proba, num_features=10
    )
    tag = "wrong_pred" if idx in wrong_idx else "top_risk"
    exp.save_to_file(f"{OUTPUT_DIR}\\lime_explanation_{i+1}_{tag}_row{idx}.html")

# Show the ranking-explanation payoff on two of the curated employees
if len(curated_idx) >= 2:
    explain_ranking_difference(curated_idx[0], curated_idx[1], agg_shap_lr, results_df)

print(f"\n{len(curated_idx)} LIME explanations saved to {OUTPUT_DIR}")


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
fairness_report_df.to_csv(f"{OUTPUT_DIR}\\fairness_audit_report.csv", index=False)
print(f"\nFairness audit report saved: {OUTPUT_DIR}\\fairness_audit_report.csv")

# --- Deliverable: fairness visualization ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (attr_name, sensitive) in zip(axes, sensitive_attributes.items()):
    mf = MetricFrame(metrics={"selection_rate": selection_rate},
                      y_true=y_test_reset, y_pred=y_pred_lr, sensitive_features=sensitive)
    mf.by_group["selection_rate"].plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title(f"Selection Rate by {attr_name}")
    ax.set_ylabel("Selection rate (flagged as at-risk)")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}\\fairness_selection_rates.png", dpi=130)
plt.close()


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
mitigation_summary.to_csv(f"{OUTPUT_DIR}\\mitigation_comparison.csv", index=False)
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
print(f"  - {OUTPUT_DIR}\\lime_explanation_*.html (8 files)")
print(f"  - {OUTPUT_DIR}\\fairness_audit_report.csv, fairness_selection_rates.png")
print(f"  - {OUTPUT_DIR}\\mitigation_comparison.csv")
