from fastapi.testclient import TestClient
from src.server import app

client = TestClient(app)

def test_reset_endpoint():
    response = client.post("/reset")
    assert response.status_code == 200
    data = response.json()
    assert "observation" in data
    assert "system_logs" in data["observation"]

def test_state_endpoint():
    response = client.get("/state")
    assert response.status_code == 200
    data = response.json()
    assert data["meters_online"] == 99

def test_full_lifecycle():
    # 1. Reset
    client.post("/reset")
    
    # 2. Complete Task 1
    resp_1 = client.post("/step", json={"command": "RESTART", "target_meter_id": "METER-042"})
    assert resp_1.json()["reward"] == 1.0
    
    # 3. Complete Task 2
    resp_2 = client.post("/step", json={"command": "PATCH", "payload": "v2.0"})
    assert resp_2.json()["reward"] == 1.0
    
    # 4. Complete Task 3
    resp_3 = client.post("/step", json={"command": "REROUTE", "target_meter_id": "ALL"})
    assert resp_3.json()["reward"] == 1.0
    assert resp_3.json()["done"] is True