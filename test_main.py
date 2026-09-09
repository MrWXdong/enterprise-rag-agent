import pytest
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


class TestHealthEndpoint:
    """测试 /health 健康检查接口"""

    def test_health_endpoint_returns_ok_status(self):
        """测试健康检查接口返回正确状态"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "enterprise_rag_agent"}

    def test_health_endpoint_returns_correct_content_type(self):
        """测试健康检查接口返回正确的内容类型"""
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"

    def test_health_endpoint_response_structure(self):
        """测试健康检查接口响应结构"""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert isinstance(data["status"], str)
        assert isinstance(data["service"], str)