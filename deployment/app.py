from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "output.1" / "best_attrition_model.pkl"

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="API for predicting voluntary employee attrition risk.",
    version="1.0.0",
)

@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

class EmployeeData(BaseModel):
    # fields...
    pass

@app.get("/")
def root():
    return {
        "message": "Employee Attrition Prediction API is running",
        "endpoint": "/predict",
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL_PATH.exists(),
    }

@app.post("/predict")
def predict(data: EmployeeData):
    try:
        model = load_model()
        input_data = pd.DataFrame([data.model_dump()])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        return {
            "prediction": int(prediction),
            "label": "Likely to Leave" if prediction == 1 else "Likely to Stay",
            "probability": round(float(probability), 4),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
