from fastapi.testclient import TestClient
from src.predict import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "engine_rpm": 3200.0,
        "engine_temp": 110.0,
        "vibration_level": 4.2
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "failure_probability" in data
    assert "status" in data