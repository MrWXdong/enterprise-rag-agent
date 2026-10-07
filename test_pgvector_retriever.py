from pgvector import Vector

from config import settings
import pgvector_retriever


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed = []
        self.executemany_args = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def executemany(self, sql, params):
        self.executemany_args = (sql, params)

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, cursor):
        self.cursor_instance = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def cursor(self):
        return self.cursor_instance


class TestStoreChunks:
    """测试 chunk 向量写入"""

    def test_store_chunks_normal(self, monkeypatch):
        cursor = FakeCursor()
        connection = FakeConnection(cursor)

        monkeypatch.setattr(
            pgvector_retriever,
            "embed_texts",
            lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
        )
        monkeypatch.setattr(
            pgvector_retriever,
            "get_connection",
            lambda: connection,
        )

        chunks = [
            {
                "chunk_id": "doc-1_chunk_000000",
                "document_id": "doc-1",
                "chunk_index": 0,
                "content": "住宿费用标准",
                "chapter": "第一章 公司制度",
                "heading": "第四条 住宿费用标准",
                "heading_type": "article",
            }
        ]

        result = pgvector_retriever.store_chunks(chunks)

        assert result == 1
        assert len(cursor.executed) == 1

        delete_sql, delete_params = cursor.executed[0]
        assert "DELETE FROM document_chunks" in delete_sql
        assert delete_params == ("doc-1",)

        assert cursor.executemany_args is not None
        _, rows = cursor.executemany_args

        assert len(rows) == 1
        assert isinstance(rows[0][-1], Vector)
        assert rows[0][7] == settings.embedding_model

    def test_store_chunks_empty_returns_zero(self):
        assert pgvector_retriever.store_chunks([]) == 0


class TestSearchChunks:
    """测试 PgVector Top-K 检索"""

    def test_search_chunks_normal(self, monkeypatch):
        rows = [
            (
                "doc-1_chunk_000000",
                "doc-1",
                0,
                "一类城市每晚最高报销六百元。",
                "第一章 公司制度",
                "第四条 住宿费用标准",
                "article",
                settings.embedding_model,
                0.82,
            )
        ]

        cursor = FakeCursor(rows=rows)
        connection = FakeConnection(cursor)

        monkeypatch.setattr(
            pgvector_retriever,
            "embed_text",
            lambda query: [0.1, 0.2, 0.3],
        )
        monkeypatch.setattr(
            pgvector_retriever,
            "get_connection",
            lambda: connection,
        )

        results = pgvector_retriever.search_chunks(
            "一类城市住宿费多少？",
            top_k=1,
        )

        assert len(results) == 1
        assert results[0]["chunk_id"] == "doc-1_chunk_000000"
        assert results[0]["heading"] == "第四条 住宿费用标准"
        assert results[0]["score"] == 0.82

        _, params = cursor.executed[0]
        assert isinstance(params[0], Vector)
        assert isinstance(params[1], Vector)
        assert params[2] == 1

    def test_search_chunks_invalid_top_k(self):
        try:
            pgvector_retriever.search_chunks("问题", top_k=0)
            assert False, "top_k=0 应该抛出 ValueError"
        except ValueError:
            pass
