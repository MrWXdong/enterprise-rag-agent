import pytest

import memory_retriever


class TestSearchChunks:
    """测试内存版 Top-K 检索"""

    def test_empty_chunks_returns_empty_list(self):
        assert memory_retriever.search_chunks([], "问题") == []

    def test_invalid_top_k_raises_error(self):
        with pytest.raises(ValueError):
            memory_retriever.search_chunks(
                [{"content": "测试"}],
                "问题",
                top_k=0,
            )

    def test_search_returns_most_similar_chunk_first(
        self,
        monkeypatch,
    ):
        chunks = [
            {
                "chunk_id": "chunk-1",
                "content": "住宿费用标准",
            },
            {
                "chunk_id": "chunk-2",
                "content": "电脑采购流程",
            },
        ]

        def fake_embed_texts(texts):
            return [
                [1.0, 0.0],
                [0.0, 1.0],
            ]

        def fake_embed_text(text):
            return [1.0, 0.0]

        monkeypatch.setattr(
            memory_retriever,
            "embed_texts",
            fake_embed_texts,
        )
        monkeypatch.setattr(
            memory_retriever,
            "embed_text",
            fake_embed_text,
        )

        results = memory_retriever.search_chunks(
            chunks,
            "住宿费怎么报销",
            top_k=2,
        )

        assert len(results) == 2
        assert results[0]["chunk_id"] == "chunk-1"
        assert results[0]["score"] == 1.0
        assert results[1]["chunk_id"] == "chunk-2"
        assert results[1]["score"] == 0.0

    def test_top_k_limits_results(self, monkeypatch):
        chunks = [{"chunk_id": f"chunk-{i}", "content": f"文本{i}"} for i in range(5)]

        def fake_embed_texts(texts):
            return [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.7, 0.3],
                [0.6, 0.4],
            ]

        monkeypatch.setattr(
            memory_retriever,
            "embed_texts",
            fake_embed_texts,
        )
        monkeypatch.setattr(
            memory_retriever,
            "embed_text",
            lambda text: [1.0, 0.0],
        )

        results = memory_retriever.search_chunks(
            chunks,
            "问题",
            top_k=3,
        )

        assert len(results) == 3
        assert [item["chunk_id"] for item in results] == [
            "chunk-0",
            "chunk-1",
            "chunk-2",
        ]
