from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "output.1" / "best_attrition_model.pkl"


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model from {MODEL_PATH}: {e}")


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="API for predicting voluntary employee attrition risk.",
    version="1.0.0"
)


# ---------------------------------------------------------
# Request schema
# ---------------------------------------------------------

class EmployeeData(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: float
    Department: str
    DistanceFromHome: float
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: float
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: float
    MonthlyRate: float
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: float
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: float
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: float
    YearsInCurrentRole: float
    YearsSinceLastPromotion: float
    YearsWithCurrManager: float
    recruitment_source: str
    tenure_bucket: str
    is_new_hire: int
    age_group: str
    income_per_experience_year: float
    satisfaction_composite: float
    hired_on_weekend: int
    recruitment_source: str


# ---------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Employee Attrition Prediction API is running",
        "endpoint": "/predict"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
def predict(data: EmployeeData):
    try:
        input_data = pd.DataFrame([data.model_dump()])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        return {
            "prediction": int(prediction),
            "label": "Likely to Leave" if prediction == 1 else "Likely to Stay",
            "probability": round(float(probability), 4)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
