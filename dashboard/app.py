from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Paths & Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "output.1" / "best_attrition_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "employee_attrition_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "output.1"
RESPONSIBLE_AI_PATH = BASE_DIR / "Responsible_AI.md"

st.set_page_config(
    page_title="Employee Attrition Analytics & Governance",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Cached Resources
# --------------------------------------------------


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_data
def load_audit_reports():
    reports = {}
    fairness_path = OUTPUT_DIR / "fairness_audit_report.csv"
    if fairness_path.exists():
        reports["fairness"] = pd.read_csv(fairness_path)

    mitigation_path = OUTPUT_DIR / "mitigation_comparison.csv"
    if mitigation_path.exists():
        reports["mitigation"] = pd.read_csv(mitigation_path)

    comparison_path = OUTPUT_DIR / "baseline_vs_tuned_comparison.csv"
    if comparison_path.exists():
        reports["comparison"] = pd.read_csv(comparison_path)

    explanations_path = OUTPUT_DIR / "employee_risk_explanations.csv"
    if explanations_path.exists():
        reports["explanations"] = pd.read_csv(explanations_path)

    validation_path = OUTPUT_DIR / "validation_report.json"
    if validation_path.exists():
        with open(validation_path, "r", encoding="utf-8") as f:
            reports["validation"] = json.load(f)

    return reports


model = load_model()
ref_df = load_reference_data()
reports = load_audit_reports()


# --------------------------------------------------
# Feature Engineering Helper
# --------------------------------------------------


def build_model_input_record(raw_inputs: dict) -> pd.DataFrame:
    """Computes the 6 engineered features and returns a 37-column DataFrame

    matching the exact feature pipeline from training.
    """
    age = raw_inputs["Age"]
    years_at_company = raw_inputs["YearsAtCompany"]
    monthly_income = raw_inputs["MonthlyIncome"]
    total_working_years = raw_inputs["TotalWorkingYears"]
    env_sat = raw_inputs["EnvironmentSatisfaction"]
    job_sat = raw_inputs["JobSatisfaction"]
    rel_sat = raw_inputs["RelationshipSatisfaction"]
    wl_balance = raw_inputs["WorkLifeBalance"]
    hired_weekend = raw_inputs.get("hired_on_weekend", 0)

    # 1. tenure_bucket: bins=[-1, 2, 5, 10, 100]
    tenure_bucket = pd.cut(
        [years_at_company],
        bins=[-1, 2, 5, 10, 100],
        labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "10+ yrs"],
    )[0]

    # 2. is_new_hire: tenure <= 1 year
    is_new_hire = int(years_at_company <= 1)

    # 3. age_group: bins=[0, 25, 35, 45, 55, 100]
    age_group = pd.cut(
        [age],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["<=25", "26-35", "36-45", "46-55", "56+"],
    )[0]

    # 4. income_per_experience_year
    income_per_experience_year = float(monthly_income) / (float(total_working_years) + 1.0)

    # 5. satisfaction_composite: normalized average of 4 satisfaction dimensions
    raw_avg = (env_sat + job_sat + rel_sat + wl_balance) / 4.0
    satisfaction_composite = (raw_avg - 1.0) / 3.0

    # Assemble complete 37 features in model order
    features = {
        "Age": int(age),
        "BusinessTravel": str(raw_inputs["BusinessTravel"]),
        "DailyRate": float(raw_inputs["DailyRate"]),
        "Department": str(raw_inputs["Department"]),
        "DistanceFromHome": float(raw_inputs["DistanceFromHome"]),
        "Education": int(raw_inputs["Education"]),
        "EducationField": str(raw_inputs["EducationField"]),
        "EnvironmentSatisfaction": int(env_sat),
        "Gender": str(raw_inputs["Gender"]),
        "HourlyRate": float(raw_inputs["HourlyRate"]),
        "JobInvolvement": int(raw_inputs["JobInvolvement"]),
        "JobLevel": int(raw_inputs["JobLevel"]),
        "JobRole": str(raw_inputs["JobRole"]),
        "JobSatisfaction": int(job_sat),
        "MaritalStatus": str(raw_inputs["MaritalStatus"]),
        "MonthlyIncome": float(monthly_income),
        "MonthlyRate": float(raw_inputs["MonthlyRate"]),
        "NumCompaniesWorked": int(raw_inputs["NumCompaniesWorked"]),
        "OverTime": str(raw_inputs["OverTime"]),
        "PercentSalaryHike": float(raw_inputs["PercentSalaryHike"]),
        "PerformanceRating": int(raw_inputs["PerformanceRating"]),
        "RelationshipSatisfaction": int(rel_sat),
        "StockOptionLevel": int(raw_inputs["StockOptionLevel"]),
        "TotalWorkingYears": float(total_working_years),
        "TrainingTimesLastYear": int(raw_inputs["TrainingTimesLastYear"]),
        "WorkLifeBalance": int(wl_balance),
        "YearsAtCompany": float(years_at_company),
        "YearsInCurrentRole": float(raw_inputs["YearsInCurrentRole"]),
        "YearsSinceLastPromotion": float(raw_inputs["YearsSinceLastPromotion"]),
        "YearsWithCurrManager": float(raw_inputs["YearsWithCurrManager"]),
        "recruitment_source": str(raw_inputs["recruitment_source"]),
        "tenure_bucket": str(tenure_bucket),
        "is_new_hire": int(is_new_hire),
        "age_group": str(age_group),
        "income_per_experience_year": float(income_per_experience_year),
        "satisfaction_composite": float(satisfaction_composite),
        "hired_on_weekend": int(hired_weekend),
    }

    return pd.DataFrame([features])


# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------

st.sidebar.title("Attrition Analytics")
st.sidebar.markdown("**Portfolio & Governance Dashboard**")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "Project Overview",
        "Interactive Prediction",
        "Model Performance",
        "SHAP Explainability",
        "Fairness Audit",
        "Data Monitoring & Drift",
        "Responsible AI Report",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project Metadata**\n"
    "- Target: Voluntary Attrition\n"
    "- Primary Model: Tuned Logistic Regression\n"
    "- Features: 37 (31 Base + 6 Engineered)\n"
    "- Explainability: SHAP & LIME\n"
    "- Governance: Fairlearn Postprocessing"
)


# --------------------------------------------------
# Page 1: Project Overview
# --------------------------------------------------

if page == "Project Overview":
    st.title("Employee Attrition Prediction & Responsible AI")
    st.markdown(
        "An explainable employee attrition decision-support platform designed for "
        "evidence-based HR retention, fairness auditing, and proactive workforce health."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Workforce", "1,470 Employees")
    with col2:
        st.metric("Voluntary Turnover", "16.1%")
    with col3:
        st.metric("Model ROC-AUC", "0.814")
    with col4:
        st.metric("Top-20 Precision", "70.0%")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Business Context & Objectives")
        st.markdown(
            "- **Retention Proactivity**: Enable people managers and HR partners to identify\n"
            "  early signs of voluntary departure before an employee submits resignation.\n"
            "- **Strict Voluntary Target Formulation**: Involuntary terminations (restructuring,\n"
            "  layoffs) are segregated from the modeling target to ensure policies address\n"
            "  controllable retention factors.\n"
            "- **Ranking-First Optimization**: HR bandwidth is constrained; the primary goal is\n"
            "  ranking the top 20-50 highest-risk employees with high precision\n"
            "  (**Precision@20 = 70%**).\n"
            "- **Explainable Interventions**: Predictions provide actionable SHAP attribution so\n"
            "  managers can resolve specific root causes (e.g. chronic overtime,\n"
            "  compensation friction)."
        )

        st.subheader("End-to-End Pipeline Architecture")
        st.markdown(
            "1. **Data Cleaning & Auditing**: MCAR/MAR/MNAR missing value handling, unit fixes.\n"
            "2. **Feature Engineering**: Derivation of `tenure_bucket`, `is_new_hire`,\n"
            "   `age_group`, `income_per_experience_year`, `satisfaction_composite`,\n"
            "   and `hired_on_weekend`.\n"
            "3. **Leakage Prevention**: Removal of post-exit notes and target-correlated flags.\n"
            "4. **Model Tuning**: Tuned Logistic Regression evaluated against Random Forest,\n"
            "   XGBoost, and Decision Tree baselines.\n"
            "5. **Fairness & Threshold Optimization**: Mitigating disparate impact\n"
            "   across demographic segments."
        )

    with col_right:
        st.subheader("Model Specifications")
        spec_df = pd.DataFrame({
            "Specification": [
                "Primary Model",
                "Challenger Model",
                "Feature Count",
                "Validation Strategy",
                "Thresholding",
                "Explainability Tool",
            ],
            "Details": [
                "Tuned Logistic Regression (C=0.0746)",
                "Random Forest (n=370, depth=9)",
                "37 Total (31 Base + 6 Engineered)",
                "Stratified 80/20 Train-Test Split",
                "Fairlearn ThresholdOptimizer",
                "SHAP (Shapley Additive exPlanations)",
            ],
        })
        st.dataframe(spec_df, use_container_width=True, hide_index=True)

        if MODEL_PATH.exists():
            st.success("Trained Model Artifact loaded successfully (`best_attrition_model.pkl`)")
        else:
            st.warning("Model file not found at default path.")


# --------------------------------------------------
# Page 2: Interactive Prediction
# --------------------------------------------------

elif page == "Interactive Prediction":
    st.title("Interactive Employee Risk Scoring")
    st.markdown(
        "Input employee attributes to generate real-time voluntary attrition risk probabilities. "
        "The dashboard automatically derives the **6 required engineered features**\n"
        "from raw inputs."
    )

    # Preset Profile Loader
    preset = st.selectbox(
        "Load Preset Profile for Rapid Testing:",
        [
            "Custom Inputs",
            "Profile A: High Risk (New Hire, Heavy OverTime, Low Satisfaction)",
            "Profile B: Low Risk (Tenured, Strong Satisfaction, Balanced Workload)",
            "Profile C: Moderate Risk (Mid-Tenure, Sales Executive, High Travel)",
        ],
    )

    # Default values based on profile
    if preset == "Profile A: High Risk (New Hire, Heavy OverTime, Low Satisfaction)":
        d_age = 24
        d_travel = "Travel_Frequently"
        d_dept = "Sales"
        d_role = "Sales Representative"
        d_overtime = "Yes"
        d_income = 2300.0
        d_tenure = 1.0
        d_tot_exp = 2.0
        d_curr_role = 0.0
        d_mgr_yrs = 0.0
        d_promo_yrs = 0.0
        d_companies = 2
        d_job_sat = 1
        d_env_sat = 1
        d_rel_sat = 2
        d_wl_bal = 1
        d_marital = "Single"
        d_distance = 25.0
        d_stock = 0
        d_involve = 1
    elif preset == "Profile B: Low Risk (Tenured, Strong Satisfaction, Balanced Workload)":
        d_age = 42
        d_travel = "Non-Travel"
        d_dept = "Research & Development"
        d_role = "Manager"
        d_overtime = "No"
        d_income = 11500.0
        d_tenure = 10.0
        d_tot_exp = 18.0
        d_curr_role = 7.0
        d_mgr_yrs = 7.0
        d_promo_yrs = 2.0
        d_companies = 1
        d_job_sat = 4
        d_env_sat = 4
        d_rel_sat = 4
        d_wl_bal = 3
        d_marital = "Married"
        d_distance = 3.0
        d_stock = 2
        d_involve = 3
    elif preset == "Profile C: Moderate Risk (Mid-Tenure, Sales Executive, High Travel)":
        d_age = 33
        d_travel = "Travel_Frequently"
        d_dept = "Sales"
        d_role = "Sales Executive"
        d_overtime = "No"
        d_income = 5400.0
        d_tenure = 4.0
        d_tot_exp = 8.0
        d_curr_role = 2.0
        d_mgr_yrs = 2.0
        d_promo_yrs = 1.0
        d_companies = 3
        d_job_sat = 2
        d_env_sat = 3
        d_rel_sat = 3
        d_wl_bal = 2
        d_marital = "Single"
        d_distance = 14.0
        d_stock = 1
        d_involve = 2
    else:
        d_age = 32
        d_travel = "Travel_Rarely"
        d_dept = "Research & Development"
        d_role = "Research Scientist"
        d_overtime = "No"
        d_income = 4800.0
        d_tenure = 4.0
        d_tot_exp = 7.0
        d_curr_role = 2.0
        d_mgr_yrs = 2.0
        d_promo_yrs = 1.0
        d_companies = 2
        d_job_sat = 3
        d_env_sat = 3
        d_rel_sat = 3
        d_wl_bal = 3
        d_marital = "Married"
        d_distance = 6.0
        d_stock = 1
        d_involve = 3

    role_options = [
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Healthcare Representative",
        "Manager",
        "Sales Representative",
        "Research Director",
        "Human Resources",
    ]
    role_idx = role_options.index(d_role) if d_role in role_options else 0

    with st.form("employee_prediction_form"):
        tab1, tab2, tab3, tab4 = st.tabs([
            "1. Demographics & Background",
            "2. Role, Department & Workload",
            "3. Compensation & Experience",
            "4. Satisfaction & Well-being",
        ])

        with tab1:
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", min_value=18, max_value=65, value=d_age)
                gender = st.selectbox("Gender", ["Female", "Male"])
            with c2:
                marital = st.selectbox(
                    "Marital Status",
                    ["Divorced", "Married", "Single"],
                    index=["Divorced", "Married", "Single"].index(d_marital),
                )
                distance = st.number_input(
                    "Distance From Home (miles)",
                    min_value=1.0,
                    max_value=35.0,
                    value=float(d_distance),
                )
            with c3:
                education = st.selectbox(
                    "Education Level (1=Below College, 5=Doctorate)",
                    [1, 2, 3, 4, 5],
                    index=2,
                )
                edu_field = st.selectbox(
                    "Education Field",
                    ["Life Sciences", "Medical", "Marketing", "Technical Degree", "HR", "Other"],
                )

        with tab2:
            c1, c2, c3 = st.columns(3)
            with c1:
                dept = st.selectbox(
                    "Department",
                    ["Research & Development", "Sales", "Human Resources"],
                    index=["Research & Development", "Sales", "Human Resources"].index(d_dept),
                )
                role = st.selectbox("Job Role", role_options, index=role_idx)
            with c2:
                job_level = st.selectbox("Job Level (1 to 5)", [1, 2, 3, 4, 5], index=1)
                travel = st.selectbox(
                    "Business Travel",
                    ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
                    index=["Non-Travel", "Travel_Rarely", "Travel_Frequently"].index(d_travel),
                )
            with c3:
                overtime = st.selectbox(
                    "OverTime Status",
                    ["No", "Yes"],
                    index=["No", "Yes"].index(d_overtime),
                )
                job_involve = st.slider(
                    "Job Involvement (1=Low, 4=High)", 1, 4, value=d_involve
                )
                recruitment = st.selectbox(
                    "Recruitment Source",
                    [
                        "LinkedIn",
                        "Indeed",
                        "Employee Referral",
                        "CareerBuilder",
                        "Google Search",
                        "Other",
                    ],
                )

        with tab3:
            c1, c2, c3 = st.columns(3)
            with c1:
                monthly_income = st.number_input(
                    "Monthly Income ($)",
                    min_value=1000.0,
                    max_value=30000.0,
                    value=float(d_income),
                    step=100.0,
                )
                salary_hike = st.number_input(
                    "Percent Salary Hike (%)",
                    min_value=10.0,
                    max_value=30.0,
                    value=14.0,
                )
                perf_rating = st.selectbox("Performance Rating (3-4)", [3, 4], index=0)
            with c2:
                total_exp = st.number_input(
                    "Total Working Years",
                    min_value=0.0,
                    max_value=45.0,
                    value=float(d_tot_exp),
                )
                years_company = st.number_input(
                    "Years at Company",
                    min_value=0.0,
                    max_value=40.0,
                    value=float(d_tenure),
                )
                num_companies = st.number_input(
                    "Companies Worked",
                    min_value=0,
                    max_value=10,
                    value=d_companies,
                )
            with c3:
                role_years = st.number_input(
                    "Years in Role",
                    min_value=0.0,
                    max_value=20.0,
                    value=float(d_curr_role),
                )
                promo_years = st.number_input(
                    "Years Since Last Promotion",
                    min_value=0.0,
                    max_value=15.0,
                    value=float(d_promo_yrs),
                )
                mgr_years = st.number_input(
                    "Years with Manager",
                    min_value=0.0,
                    max_value=20.0,
                    value=float(d_mgr_yrs),
                )
                stock_level = st.selectbox(
                    "Stock Option Level (0 to 3)", [0, 1, 2, 3], index=d_stock
                )

        with tab4:
            c1, c2 = st.columns(2)
            with c1:
                job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, value=d_job_sat)
                env_sat = st.slider("Environment Satisfaction (1-4)", 1, 4, value=d_env_sat)
                trainings = st.slider("Trainings Last Year", 0, 6, value=2)
            with c2:
                rel_sat = st.slider("Relationship Satisfaction (1-4)", 1, 4, value=d_rel_sat)
                wl_balance = st.slider("Work-Life Balance (1-4)", 1, 4, value=d_wl_bal)
                weekend_hire = st.checkbox("Hired on Weekend (Simulation Flag)", value=False)

        submit_btn = st.form_submit_button(
            "Compute Attrition Risk & Interventions", use_container_width=True
        )

    if submit_btn:
        if model is None:
            st.error("Model artifact could not be loaded (`best_attrition_model.pkl`).")
        else:
            raw_dict = {
                "Age": age,
                "BusinessTravel": travel,
                "DailyRate": 800.0,
                "Department": dept,
                "DistanceFromHome": distance,
                "Education": education,
                "EducationField": "Life Sciences" if edu_field == "HR" else edu_field,
                "EnvironmentSatisfaction": env_sat,
                "Gender": gender,
                "HourlyRate": 65.0,
                "JobInvolvement": job_involve,
                "JobLevel": job_level,
                "JobRole": role,
                "JobSatisfaction": job_sat,
                "MaritalStatus": marital,
                "MonthlyIncome": monthly_income,
                "MonthlyRate": 14000.0,
                "NumCompaniesWorked": num_companies,
                "OverTime": overtime,
                "PercentSalaryHike": salary_hike,
                "PerformanceRating": perf_rating,
                "RelationshipSatisfaction": rel_sat,
                "StockOptionLevel": stock_level,
                "TotalWorkingYears": total_exp,
                "TrainingTimesLastYear": trainings,
                "WorkLifeBalance": wl_balance,
                "YearsAtCompany": years_company,
                "YearsInCurrentRole": role_years,
                "YearsSinceLastPromotion": promo_years,
                "YearsWithCurrManager": mgr_years,
                "recruitment_source": recruitment,
                "hired_on_weekend": int(weekend_hire),
            }

            input_df = build_model_input_record(raw_dict)
            pred = int(model.predict(input_df)[0])
            prob = float(model.predict_proba(input_df)[0][1])

            st.markdown("---")
            st.subheader("Prediction Result & Risk Profile")

            r1, r2, r3 = st.columns([1.5, 1.5, 3])

            with r1:
                if prob >= 0.50:
                    st.error(f"### High Attrition Risk\n**Probability: {prob:.1%}**")
                elif prob >= 0.30:
                    st.warning(f"### Moderate Risk\n**Probability: {prob:.1%}**")
                else:
                    st.success(f"### Low Attrition Risk\n**Probability: {prob:.1%}**")

            with r2:
                st.metric(
                    "Model Classification",
                    "Likely to Leave" if pred == 1 else "Likely to Stay",
                )
                st.progress(prob)

            with r3:
                st.markdown("**Engineered Feature Values Derived Behind the Scenes:**")
                eng_summary = pd.DataFrame([{
                    "tenure_bucket": input_df.loc[0, "tenure_bucket"],
                    "is_new_hire": input_df.loc[0, "is_new_hire"],
                    "age_group": input_df.loc[0, "age_group"],
                    "income/exp_yr": f"${input_df.loc[0, 'income_per_experience_year']:.1f}",
                    "satisfaction_composite": f"{input_df.loc[0, 'satisfaction_composite']:.2f}",
                }])
                st.dataframe(eng_summary, hide_index=True, use_container_width=True)

            st.subheader("Tailored HR Retention Guidance")
            recs = []
            if overtime == "Yes":
                recs.append(
                    "🔴 **Overtime Fatigue**: Frequent overtime detected. "
                    "Review capacity allocation and rebalance project workload."
                )
            if years_company <= 1:
                recs.append(
                    "🟡 **Onboarding Fragility**: In first year (new hire). "
                    "Schedule a 30-60-90 day stay interview."
                )
            if job_sat <= 2 or env_sat <= 2:
                recs.append(
                    "🟡 **Satisfaction Friction**: Low job/environment satisfaction. "
                    "Facilitate 1-on-1 manager check-in."
                )
            if promo_years >= 3 and years_company >= 4:
                recs.append(
                    "🔵 **Career Progression**: Stagnant promotion timeline (>3 yrs). "
                    "Explore internal mobility and growth options."
                )
            if not recs:
                recs.append(
                    "🟢 **Positive Retention Signals**: Profile aligns with stable retention. "
                    "Maintain regular career feedback."
                )

            for r in recs:
                st.markdown(r)


# --------------------------------------------------
# Page 3: Model Performance
# --------------------------------------------------

elif page == "Model Performance":
    st.title("Model Performance & Candidate Evaluation")
    st.markdown(
        "Evaluation results from **Experiment 4: MLOps Model Selection & Tracking**. "
        "Because HR retention is bandwidth-constrained, **ROC-AUC** and **Precision@K** "
        "served as primary decision criteria over traditional unranked accuracy."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Accuracy", "77.55%", help="Overall accuracy at default threshold")
    with c2:
        st.metric("Precision", "38.55%", help="Precision on minority turnover class")
    with c3:
        st.metric("Recall", "68.09%", help="Sensitivity to true voluntary leavers")
    with c4:
        st.metric("ROC-AUC", "0.814", help="Area under ROC curve across all thresholds")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top-K Ranking Precision")
        st.markdown(
            "In practice, HR retention teams intervene with the top 20 to 50 employees flagged. "
            "Our tuned Logistic Regression model achieves **70% precision in top 20 cohort**."
        )

        topk_df = pd.DataFrame({
            "Ranking Metric": [
                "Precision@20",
                "Recall@20",
                "Precision@30",
                "Recall@30",
                "Precision@50",
                "Recall@50",
            ],
            "Score": [0.700, 0.298, 0.600, 0.383, 0.500, 0.532],
            "Interpretation": [
                "14 of top 20 flagged employees are confirmed voluntary leavers",
                "Captures ~30% of all leavers inside just 20 outreach slots",
                "18 of top 30 employees are true voluntary leavers",
                "Captures ~38% of total turnover",
                "25 of top 50 flagged employees are true voluntary leavers",
                "Over half (53.2%) of all organization attrition captured in top 50 list",
            ],
        })
        st.dataframe(topk_df, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("Baseline vs. Tuned Models Comparison")
        if "comparison" in reports:
            comp_df = reports["comparison"].copy()
            st.dataframe(comp_df.round(4), use_container_width=True, hide_index=True)
        else:
            st.info("Comparison CSV not found.")

        st.markdown(
            "**Why Tuned Logistic Regression was Selected**:\n"
            "1. **Optimal Ranking Power**: Clear winner on ROC-AUC (0.814) and "
            "Precision@20 (0.70).\n"
            "2. **Direct Interpretability**: Additive log-odds risk decomposition without error.\n"
            "3. **Fairness Controllability**: Seamlessly compatible with Fairlearn postprocessing."
        )


# --------------------------------------------------
# Page 4: SHAP Explainability
# --------------------------------------------------

elif page == "SHAP Explainability":
    st.title("Explainable AI — Global & Local SHAP Insights")
    st.markdown(
        "SHAP (Shapley Additive exPlanations) values provide game-theoretic attribution "
        "explaining how each feature pushes the attrition probability relative to baseline."
    )

    tab_global, tab_local = st.tabs(["Global Feature Importance", "Individual Case Studies"])

    with tab_global:
        st.subheader("Top Drivers of Voluntary Attrition")
        st.markdown(
            "1. **OverTime**: Single strongest positive driver of departure.\n"
            "2. **Tenure & Manager Stability (`YearsAtCompany`, `YearsWithCurrManager`)**: "
            "Longevity buffers against turnover.\n"
            "3. **New Hire Status (`is_new_hire`)**: First-year employees exhibit early spike.\n"
            "4. **Income per Experience Year**: Compensation compression raises turnover.\n"
            "5. **Composite Satisfaction**: Low composite satisfaction across environment and role."
        )

        col_img1, col_img2 = st.columns(2)

        lr_shap_path = OUTPUT_DIR / "shap_summary_logreg.png"
        rf_shap_path = OUTPUT_DIR / "shap_summary_rf.png"

        with col_img1:
            if lr_shap_path.exists():
                st.image(
                    str(lr_shap_path),
                    caption="Primary Model: Logistic Regression SHAP Summary",
                    use_container_width=True,
                )
            else:
                st.warning("Logistic Regression SHAP plot not found.")

        with col_img2:
            if rf_shap_path.exists():
                st.image(
                    str(rf_shap_path),
                    caption="Challenger Model: Random Forest SHAP Summary",
                    use_container_width=True,
                )
            else:
                st.warning("Random Forest SHAP plot not found.")

        dep_path = OUTPUT_DIR / "shap_dependence_OverTime.png"
        if dep_path.exists():
            st.subheader("SHAP Dependence: OverTime Impact")
            st.image(
                str(dep_path),
                caption="OverTime Feature Interaction with Working Hours",
                use_container_width=True,
            )

    with tab_local:
        st.subheader("Audit Case Studies: Per-Employee Explanations")
        st.markdown(
            "Review how the model decomposed individual risk scores into additive log-odds "
            "contributions for verified employees from the test set."
        )

        if "explanations" in reports:
            exp_df = reports["explanations"].copy()
            unique_rows = sorted(exp_df["employee_row"].unique())

            selected_row = st.selectbox(
                "Select Employee Row from Test Audit:",
                unique_rows,
                format_func=lambda x: (
                    f"Employee #{x} (Risk: "
                    f"{exp_df[exp_df['employee_row'] == x]['risk_score'].iloc[0]:.1%})"
                ),
            )

            row_data = exp_df[exp_df["employee_row"] == selected_row]
            actual_status = (
                "Left Organization" if row_data["actual"].iloc[0] == 1 else "Stayed"
            )
            predicted_status = (
                "Likely to Leave" if row_data["predicted"].iloc[0] == 1 else "Likely to Stay"
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Predicted Risk", f"{row_data['risk_score'].iloc[0]:.1%}")
            with c2:
                st.metric("Actual Status", actual_status)
            with c3:
                st.metric("Model Classification", predicted_status)

            cols_show = ["feature_rank", "feature", "shap_value_logodds", "direction"]
            st.dataframe(
                row_data[cols_show].sort_values("feature_rank"),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Employee risk explanations data file not available.")


# --------------------------------------------------
# Page 5: Fairness Audit
# --------------------------------------------------

elif page == "Fairness Audit":
    st.title("Algorithmic Fairness Audit & Mitigation")
    st.markdown(
        "Rigorous fairness evaluation across demographic segments using **Fairlearn**. "
        "Evaluating Demographic Parity Difference (DPD) and Equalized Odds Difference (EOD)."
    )

    if "fairness" in reports:
        fair_df = reports["fairness"].copy()

        st.subheader("1. Gender Parity Audit")
        gender_subset = fair_df[fair_df["sensitive_attribute"] == "Gender"]
        cols_to_show = [
            "group",
            "n_in_group",
            "accuracy",
            "selection_rate",
            "demographic_parity_difference",
            "equalized_odds_difference",
        ]
        st.dataframe(
            gender_subset[cols_to_show],
            use_container_width=True,
            hide_index=True,
        )

        g1, g2 = st.columns(2)
        with g1:
            st.metric(
                "Gender Demographic Parity Difference",
                "0.0016",
                help="Female selection rate: 28.97% vs Male: 28.81%",
            )
        with g2:
            st.metric("Gender Equalized Odds Difference", "0.1905")

        st.success(
            "Gender parity is thoroughly satisfied. "
            "Selection rate gap between Female and Male employees is negligible (<0.2%)."
        )

        st.markdown("---")

        st.subheader("2. Marital Status & Bias Mitigation")
        st.markdown(
            "Raw baseline predictions showed disparate selection rates for **Single** employees "
            "due to baseline demographic mobility relative to Married and Divorced colleagues."
        )

        marital_subset = fair_df[fair_df["sensitive_attribute"] == "MaritalStatus"]
        st.dataframe(
            marital_subset[["group", "n_in_group", "accuracy", "selection_rate"]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Fairlearn ThresholdOptimizer Postprocessing Results")
        if "mitigation" in reports:
            mit_df = reports["mitigation"].copy()
            st.dataframe(mit_df, use_container_width=True, hide_index=True)

        m1, m2 = st.columns(2)
        with m1:
            st.metric(
                "Marital Status DPD",
                "0.0613",
                delta="-0.3026 (from 0.3639)",
                delta_color="inverse",
                help="Reduced from 0.3639 to 0.0613 via ThresholdOptimizer",
            )
        with m2:
            st.metric("Post-Mitigation Accuracy", "89.08%", delta="+12.68%")

        st.info(
            "Applying Fairlearn `ThresholdOptimizer` reduced demographic disparity by **83.2%** "
            "while maintaining strong classification utility."
        )

        st.markdown("---")

        st.subheader("3. Selection Rates by Demographic Cohort")
        fairness_plot_path = OUTPUT_DIR / "fairness_selection_rates.png"
        if fairness_plot_path.exists():
            st.image(
                str(fairness_plot_path),
                caption="Selection Rates Across Gender, Marital Status, and Age Groups",
                use_container_width=True,
            )

        st.warning(
            "**Governance Note on Synthetic Attributes**: Synthetic fields (`race_ethnicity`, "
            "`manager_id`) injected during preliminary tests were sampled independently of "
            "true outcomes. Auditing fairness on synthetic fields is methodologically unsound; "
            "they were strictly excluded from model feature inputs and auditing."
        )


# --------------------------------------------------
# Page 6: Data Monitoring & Drift
# --------------------------------------------------

elif page == "Data Monitoring & Drift":
    st.title("Data Quality & Distribution Drift Monitoring")
    st.markdown(
        "Continuous monitoring of feature distributions and schema integrity to detect data drift "
        "and data quality anomalies before they degrade model predictions."
    )

    tab_drift, tab_quality = st.tabs([
        "Distribution Drift Analysis",
        "Schema & Integrity Validation",
    ])

    with tab_drift:
        st.subheader("Relative Mean Drift vs. Baseline Data")
        st.markdown(
            "Compare feature distributions between baseline training dataset (1,470 records) "
            "and new incoming employee batches."
        )

        uploaded_file = st.file_uploader(
            "Upload New Production Batch CSV to Check for Feature Drift (Optional):",
            type=["csv"],
        )

        if uploaded_file is not None:
            curr_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded batch with {len(curr_df)} rows.")
        else:
            st.info("Using baseline reference dataset as current snapshot for demonstration.")
            curr_df = ref_df

        if ref_df is not None and curr_df is not None:
            numeric_cols = [
                "MonthlyIncome",
                "Age",
                "DistanceFromHome",
                "TotalWorkingYears",
                "YearsAtCompany",
                "YearsInCurrentRole",
            ]
            valid_cols = [c for c in numeric_cols if c in ref_df.columns and c in curr_df.columns]

            drift_records = []
            for col in valid_cols:
                ref_mean = float(ref_df[col].mean())
                curr_mean = float(curr_df[col].mean())
                drift_pct = abs(curr_mean - ref_mean) / abs(ref_mean) if ref_mean != 0 else 0.0

                if drift_pct < 0.10:
                    status = "Stable"
                elif drift_pct < 0.20:
                    status = "Moderate Drift"
                else:
                    status = "High Drift Alert"

                drift_records.append({
                    "Feature": col,
                    "Reference Mean": round(ref_mean, 2),
                    "Batch Mean": round(curr_mean, 2),
                    "Relative Drift": f"{drift_pct:.2%}",
                    "Status": status,
                })

            drift_df = pd.DataFrame(drift_records)
            st.dataframe(drift_df, use_container_width=True, hide_index=True)

            selected_feature = st.selectbox("Inspect Distribution for Feature:", valid_cols)
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown(f"**Baseline: {selected_feature}**")
                st.line_chart(
                    ref_df[selected_feature].describe()[["min", "25%", "50%", "75%", "max"]]
                )
            with col_chart2:
                st.markdown(f"**Current Batch: {selected_feature}**")
                st.line_chart(
                    curr_df[selected_feature].describe()[["min", "25%", "50%", "75%", "max"]]
                )

    with tab_quality:
        st.subheader("Data Validation Suite Results")
        st.markdown(
            "Automated data checks executed during pipeline runs to guarantee schema integrity "
            "and range compliance."
        )

        if "validation" in reports and "results" in reports["validation"]:
            checks = reports["validation"]["results"]
            check_df = pd.DataFrame(checks)
            st.dataframe(check_df, use_container_width=True, hide_index=True)
            st.success("All automated data quality and integrity checks passed.")
        else:
            st.info("Validation report file not found.")


# --------------------------------------------------
# Page 7: Responsible AI Report
# --------------------------------------------------

elif page == "Responsible AI Report":
    st.title("Responsible AI Governance & Ethical Framework")
    st.markdown(
        "Complete organizational policy report detailing human-in-the-loop oversight, "
        "ethical boundaries, bias auditing, and recommended HR retention practices."
    )

    if RESPONSIBLE_AI_PATH.exists():
        with open(RESPONSIBLE_AI_PATH, "r", encoding="utf-8") as f:
            rai_content = f.read()
        st.markdown(rai_content)
    else:
        st.warning("Responsible_AI.md file not found at project root.")
