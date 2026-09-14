from fastapi.testclient import TestClient

from src.serve.main import app


client = TestClient(app)

def test_invalid_contract_is_rejected():
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Lifetime",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 94.4,
        "TotalCharges": 94.4,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    
    
def test_missing_field_is_rejected():
    response = client.post("/predict", json={"gender": "Female"})
    assert response.status_code == 422
    
    
