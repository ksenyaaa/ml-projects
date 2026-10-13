from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_positive_sentiment():

    test_data = {"text": "Этот курс просто потрясающий, я узнал очень много нового!"}

    response = client.post("/predict", json=test_data)

    assert response.status_code == 200
    
    data = response.json()
    assert data["label"] == "POSITIVE"
    assert data["confidence"] > 0.8  

def test_predict_negative_sentiment():
    test_data = {"text": "Ужасный сервис, ничего не работает."}
    response = client.post("/predict", json=test_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "NEGATIVE"