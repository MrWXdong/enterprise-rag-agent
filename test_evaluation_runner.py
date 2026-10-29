import pytest

import evaluation.runner as runner


def make_dataset():
    return {
        "version": "test-v1",
        "items": [
            {
                "id": "q001",
                "question": "测试问题",
                "relevant_chunk_ids": ["gold-chunk"],
                "category": "single_chunk",
            }
        ],
    }


def make_chunk():
    return {
        "chunk_id": "gold-chunk",
        "document_id": "doc-1",
        "chunk_index": 0,
        "content": "测试答案",
        "chapter": None,
        "heading": "测试标题",
        "heading_type": None,
        "embedding_model": "test-model",
        "score": 0.90,
    }


class TestEvaluateRetrieval:
    def test_dynamic_strategy_calls_dynamic_retriever(
        self,
        monkeypatch,
    ):
        captured = {}

        def fake_dynamic(query, **kwargs):
            captured["query"] = query
            captured.update(kwargs)
            return [make_chunk()]

        def unexpected_fixed(*args, **kwargs):
            raise AssertionError("动态策略不应调用固定检索")

        monkeypatch.setattr(
            runner,
            "embed_text",
            lambda text: [0.0],
        )
        monkeypatch.setattr(
            runner,
            "search_chunks_dynamic",
            fake_dynamic,
        )
        monkeypatch.setattr(
            runner,
            "search_chunks_with_context",
            unexpected_fixed,
        )

        result = runner.evaluate_retrieval(
            make_dataset(),
            retrieval_strategy="dynamic",
            candidate_k=7,
            max_k=3,
            min_score=0.60,
            min_relative_score=0.80,
            metric_k=2,
        )

        assert captured == {
            "query": "测试问题",
            "candidate_k": 7,
            "max_k": 3,
            "min_score": 0.60,
            "min_relative_score": 0.80,
        }
        assert result["strategy"] == "dynamic"
        assert result["candidate_k"] == 7
        assert result["max_k"] == 3
        assert result["items"][0]["retrieved_chunk_ids"] == ["gold-chunk"]

    def test_fixed_strategy_calls_fixed_retriever(
        self,
        monkeypatch,
    ):
        captured = {}

        def fake_fixed(query, **kwargs):
            captured["query"] = query
            captured.update(kwargs)
            return [make_chunk()]

        def unexpected_dynamic(*args, **kwargs):
            raise AssertionError("固定策略不应调用动态检索")

        monkeypatch.setattr(
            runner,
            "embed_text",
            lambda text: [0.0],
        )
        monkeypatch.setattr(
            runner,
            "search_chunks_with_context",
            fake_fixed,
        )
        monkeypatch.setattr(
            runner,
            "search_chunks_dynamic",
            unexpected_dynamic,
        )

        result = runner.evaluate_retrieval(
            make_dataset(),
            retrieval_strategy="fixed",
            retrieval_top_k=2,
            neighbor_window=0,
            max_chunks=2,
            metric_k=2,
        )

        assert captured == {
            "query": "测试问题",
            "top_k": 2,
            "neighbor_window": 0,
            "max_chunks": 2,
        }
        assert result["strategy"] == "fixed"

    def test_invalid_strategy_raises_error(self):
        with pytest.raises(ValueError):
            runner.evaluate_retrieval(
                make_dataset(),
                retrieval_strategy="invalid",
            )

    def test_dynamic_output_path_does_not_overwrite_fixed_result(self):
        output_path = runner.build_output_path(
            retrieval_strategy="dynamic",
            retrieval_top_k=5,
            metric_k=2,
            neighbor_window=0,
            max_chunks=5,
            candidate_k=7,
            max_k=3,
            min_score=0.60,
            min_relative_score=0.80,
        )

        assert output_path.name == (
            "dynamic_candidate7_max3_score060_relative080_metric2.json"
        )
