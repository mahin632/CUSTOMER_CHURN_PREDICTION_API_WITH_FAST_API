# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Define input schema
class CustomerData(BaseModel):
    CreditScore: float
    Geography: int  # Encoded: France=0, Germany=1, Spain=2
    Gender: int     # Encoded: Female=0, Male=1
    Age: float
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.post("/predict")
def predict(data: CustomerData):
    # Convert to numpy array
    features = np.array([[ 
        data.CreditScore, data.Geography, data.Gender,
        data.Age, data.Tenure, data.Balance,
        data.NumOfProducts, data.HasCrCard,
        data.IsActiveMember, data.EstimatedSalary
    ]])

    # Scale features
    scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": round(probability, 4)
    }
