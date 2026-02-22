from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint_exists():
    r = client.get("/health")
    assert r.status_code == 200
    assert "model_loaded" in r.json()

def test_predict_shape_or_service_unavailable():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    r = client.post("/predict", json=payload)
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        data = r.json()
        assert "predicted_class" in data
        assert len(data["probabilities"]) == 3
