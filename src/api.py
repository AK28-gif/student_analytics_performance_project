import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

app = FastAPI(
    title="Student Dropout Risk Prediction API",
    description="Real-time MLOps inference service for academic intervention",
    version="2.0.0"
)

# Load model and scaler
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model artifact not found. Run train_mlflow.py first.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

class StudentFeatures(BaseModel):
    studytime: int = Field(..., ge=1, le=4, description="Weekly study time bracket (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)")
    failures: int = Field(..., ge=0, le=4, description="Past class failures")
    absences: int = Field(..., ge=0, description="Number of school absences")
    attendance_pct: float = Field(..., ge=0.0, le=100.0, description="Attendance percentage")
    avg_internal_marks: float = Field(..., ge=0.0, le=20.0, description="Average continuous assessment mark")
    age: int = Field(..., ge=15, le=25, description="Student age")
    medu: int = Field(..., ge=0, le=4, description="Mother's education level")
    fedu: int = Field(..., ge=0, le=4, description="Father's education level")

@app.get("/")
def root():
    return {
        "message": "Student Dropout Risk Prediction API is running",
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Student Risk Prediction Engine"}

@app.post("/predict")
def predict_risk(student: StudentFeatures):
    data = pd.DataFrame([student.dict()])
    
    
    if "Logistic" in str(type(model)):
        input_data = scaler.transform(data)
    else:
        input_data = data
        
    prob = float(model.predict_proba(input_data)[0][1])
    pred = int(prob >= 0.5)
    
    return {
        "dropout_risk_prediction": "High Risk" if pred == 1 else "Normal",
        "risk_probability": round(prob, 4),
        "intervention_recommended": bool(pred == 1)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)