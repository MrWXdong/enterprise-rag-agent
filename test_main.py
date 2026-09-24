from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestRoot:
    def test_root_api(self):
        response =client.get("/")
        data = response.json()
        assert response.status_code == 200
        assert data["message"] == "企业级智能知识库项目已启动"
        assert response.headers["Content-Type"] == "application/json"
        assert "message" in data
        assert isinstance(data["message"], str)
class TestHealth:
    def test_health_api(self):
        response = client.get("/health")
        data=response.json()
        assert response.status_code == 200
        assert data =={"status":"ok","service":"enterprise_rag_agent"}
        assert response.headers["Content-Type"] == "application/json"
        assert "status" in data
        assert "service" in data
        assert isinstance(data["status"],str)
        assert isinstance(data["service"],str)