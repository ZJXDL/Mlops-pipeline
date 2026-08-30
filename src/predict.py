from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the model once when the app starts
model = joblib.load("models/model.joblib")
app = FastAPI(title="Vehicle Maintenance API")

# Define the data structure we expect from users
class TelemetryData(BaseModel):
    engine_rpm: float
    engine_temp: float
    vibration_level: float

@app.post("/predict")
def predict_failure(data: TelemetryData):
    # Convert incoming JSON to a DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Make prediction (0 = Healthy, 1 = Failure)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    return {
        "prediction": int(prediction),
        "failure_probability": float(probability),
        "status": "Needs Maintenance" if prediction == 1 else "Healthy"
    }