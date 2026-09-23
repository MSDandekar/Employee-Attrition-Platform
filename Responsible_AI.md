# Responsible AI Report — Employee Attrition Platform

## 1. System Purpose & Intended Use

The **Employee Attrition Prediction Platform** is designed to identify statistical voluntary attrition risk patterns and provide explainable, human-centered decision support for Human Resources teams and people managers.

- **Primary Goal**: Facilitate proactive employee retention, early intervention (workload balancing, career coaching, compensation adjustment), and organizational health monitoring.
- **Explicit Prohibition**: The model is **strictly a decision-support assistant** and must **never** be used for automated or unsupervised adverse employment actions (e.g., termination, demotion, denial of promotion, disciplinary measures, or layoff targeting).

---

## 2. Human-in-the-Loop Oversight

All model outputs—including predicted risk probabilities, rankings, and feature contributions—must be interpreted in context by qualified HR professionals and people partners:

- **No Automated Actions**: Attrition probability flags trigger a review, not an outcome.
- **Contextual Triangulation**: HR must cross-reference model signals with qualitative factors (e.g., recent life events, restructuring, project milestones) that numerical features cannot capture.
- **Right to Contest**: Employees subject to retention interventions retain the right to engage transparently with HR regarding workload and career development without prejudice.

---

## 3. Fairness Audit & Bias Evaluation

Fairness was systematically evaluated across real demographic attributes in accordance with Fairlearn standards:

| Protected Group | Sensitive Attribute | Initial Metric (DPD) | Post-Mitigation (DPD) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Gender** | Female vs. Male | 0.0016 (DPD) | 0.0016 | Demographic parity satisfied; selection rates are virtually identical (~28.9% vs ~28.8%). |
| **Marital Status** | Divorced / Married / Single | 0.3639 (DPD) | **0.0613 (DPD)** | Single employees had higher raw risk; Fairlearn `ThresholdOptimizer` successfully mitigated disparate impact from 0.3639 to 0.0613 while preserving 89.1% accuracy. |
| **Age Group** | <=25, 26-35, 36-45, 46-55, 56+ | 0.5556 (DPD) | Monitored | Younger cohorts (<=25) exhibit naturally higher voluntary mobility early in career. |

### Excluded Synthetic Variables
Synthetic fields injected for simulation (such as `race_ethnicity` and `manager_id`) were sampled independently of true HR outcomes in exploratory experiments. Because evaluating fairness on synthetic or disconnected attributes yields ungrounded metrics, they were strictly excluded from model feature inputs and production auditing.

---

## 4. Privacy & Confidentiality

- **Minimal Necessary Data**: Only job-relevant HR metrics are utilized. Personally identifiable attributes (Employee ID, exact date of birth, home address, social security details) are stripped before model ingestion.
- **Role-Based Access Control (RBAC)**: Only credentialed HR business partners have access to employee risk scores and SHAP explainability breakdowns.
- **Secure Processing**: In-flight and at-rest encryption must protect all tabular employee records. Model weights and prediction logs reside in isolated, audited cloud environments.

---

## 5. Consent & Data Governance

- **Corporate Transparency**: Employees should receive transparent notice regarding organizational retention analytics and workforce trend modeling.
- **Auditability**: All model retraining cycles, feature definitions, and pipeline modifications are tracked via DVC and MLflow for reproducibility and regulatory compliance.
- **No Fabricated Records**: Synthetic data enrichment is marked clearly and segregated from historical workforce records.

---

## 6. Model Limitations & Epistemic Uncertainty

- **Statistical Correlation vs. Causality**: High attrition probability signifies statistical correlation with past turnover patterns; it does not indicate a causal certainty that an employee intends to resign.
- **External Validity**: The model is calibrated on specific historical workforce dynamics. Reorganization, economic downturns, industry shifts, or changes in remote work policy alter baseline attrition dynamics and require recalibration.
- **Voluntary vs. Involuntary Distinction**: The model is exclusively trained to predict **voluntary** resignation. Layoffs, organizational downsizings, and terminations for cause are explicitly filtered out to avoid corrupting retention insights with involuntary exits.

---

## 7. Sample Size & Group Reliability

- **Small Sub-Cohort Caution**: Sub-cohorts with small representation (e.g., Age 56+ with $n=9$ in test partitions) produce wide confidence intervals for fairness and error rates.
- **Decision Precaution**: Evaluators must avoid drawing broad policy conclusions from micro-cohort metrics without statistical significance testing.

---

## 8. Continuous Monitoring & Drift Detection

Production deployments must monitor:
1. **Feature Drift**: Monthly relative drift checks on key drivers (`MonthlyIncome`, `TotalWorkingYears`, `YearsAtCompany`).
2. **Target Drift**: Monitoring company-wide turnover rates against baseline (16.1%).
3. **Model Performance Decay**: Precision@K and ROC-AUC re-evaluation against actual exit interview records.
4. **Data Integrity**: Automated schema and missingness validation (as executed in the pipeline data validation suite).

---

## 9. Recommended Constructive HR Interventions

When an employee appears in high-risk ranking tiers (e.g., Top 20 / Top 50 risk lists), HR and managers should prioritize positive, constructive retention interventions:

- **Workload & Overtime Review**: Mitigation of chronic overtime (identified as a top SHAP driver of attrition).
- **Manager Alignment & Check-Ins**: Facilitation of stay interviews and 1-on-1 dialogue with leadership.
- **Compensation & Growth Realignment**: Benchmarking compensation for employees whose `income_per_experience_year` lags peer cohorts.
- **Career Pathing & Upskilling**: Targeted internal mobility opportunities and skill development for employees experiencing promotion stagnation.
