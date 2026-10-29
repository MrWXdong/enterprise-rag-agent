from fastapi.testclient import TestClient
from main import app
import main

client = TestClient(app)


class TestRoot:
    def test_root_api(self):
        response = client.get("/")
        data = response.json()
        assert response.status_code == 200
        assert data["message"] == "企业级智能知识库项目已启动"
        assert response.headers["Content-Type"] == "application/json"
        assert "message" in data
        assert isinstance(data["message"], str)


class TestHealth:
    def test_health_api(self):
        response = client.get("/health")
        data = response.json()
        assert response.status_code == 200
        assert data == {"status": "ok", "service": "enterprise_rag_agent"}
        assert response.headers["Content-Type"] == "application/json"
        assert "status" in data
        assert "service" in data
        assert isinstance(data["status"], str)
        assert isinstance(data["service"], str)


class TestChat:
    def test_no_answer_returns_refusal_without_calling_llm(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            main.settings,
            "llm_api_key",
            "test-key",
        )
        monkeypatch.setattr(
            main,
            "search_chunks_dynamic",
            lambda query, **kwargs: [],
        )

        def unexpected_openai(*args, **kwargs):
            raise AssertionError("无答案时不应调用大模型")

        monkeypatch.setattr(
            main,
            "OpenAI",
            unexpected_openai,
        )

        response = client.post(
            "/chat",
            json={"question": "知识库中不存在的问题"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "answer": "知识库中没有足够依据回答该问题。",
            "sources": [],
        }
