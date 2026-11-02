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


def make_chunk(
    chunk_id="gold-chunk",
    content="测试答案",
    score=0.90,
    rerank_score=None,
    rerank_probability=None,
):
    chunk = {
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "chunk_index": 0,
        "content": content,
        "chapter": None,
        "heading": "测试标题",
        "heading_type": None,
        "embedding_model": "test-model",
        "score": score,
    }

    if rerank_score is not None:
        chunk["rerank_score"] = rerank_score

    if rerank_probability is not None:
        chunk["rerank_probability"] = rerank_probability

    return chunk


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
            "search_chunks",
            lambda query, top_k: [make_chunk()],
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
            "search_chunks",
            lambda query, top_k: [make_chunk()],
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

    def test_dynamic_rerank_strategy_calls_reranked_retriever(
        self,
        monkeypatch,
    ):
        captured = {}

        def fake_rerank(query, chunks):
            captured["query"] = query
            captured["chunks"] = chunks
            return [
                {
                    **chunk,
                    "retrieval_score": chunk["score"],
                    "rerank_score": 2.5,
                    "rerank_probability": 0.92,
                }
                for chunk in chunks
            ]

        monkeypatch.setattr(
            runner,
            "embed_text",
            lambda text: [0.0],
        )
        monkeypatch.setattr(
            runner,
            "search_chunks",
            lambda query, top_k: [make_chunk()],
        )
        monkeypatch.setattr(
            runner,
            "rerank_chunks",
            fake_rerank,
        )

        result = runner.evaluate_retrieval(
            make_dataset(),
            retrieval_strategy="dynamic_rerank",
            candidate_k=5,
            max_k=2,
            min_probability=0.50,
            min_relative_probability=0.80,
            metric_k=2,
        )

        assert captured["query"] == "测试问题"
        assert len(captured["chunks"]) == 1
        assert result["strategy"] == "dynamic_rerank"
        assert result["min_probability"] == 0.50
        assert result["min_relative_probability"] == 0.80
        assert result["items"][0]["retrieved_retrieval_scores"] == [0.9]
        assert result["items"][0]["retrieved_rerank_scores"] == [2.5]
        assert result["items"][0][
            "retrieved_rerank_probabilities"
        ] == [0.92]
        assert result["items"][0]["candidate_rerank_scores"] == [2.5]
        assert result["items"][0][
            "candidate_rerank_probabilities"
        ] == [0.92]

    def test_rerank_metrics_use_rerank_order(self, monkeypatch):
        dataset = {
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
        reranked_chunks = [
            make_chunk(
                "gold-chunk",
                score=0.20,
                rerank_score=3.0,
                rerank_probability=0.95,
            ),
            make_chunk(
                "noise-chunk",
                score=0.90,
                rerank_score=1.0,
                rerank_probability=0.73,
            ),
        ]

        monkeypatch.setattr(
            runner,
            "embed_text",
            lambda text: [0.0],
        )
        monkeypatch.setattr(
            runner,
            "search_chunks",
            lambda query, top_k: reranked_chunks,
        )
        monkeypatch.setattr(
            runner,
            "rerank_chunks",
            lambda query, chunks: reranked_chunks,
        )

        result = runner.evaluate_retrieval(
            dataset,
            retrieval_strategy="dynamic_rerank",
            candidate_k=2,
            max_k=2,
            metric_k=2,
        )

        item = result["items"][0]

        assert item["retrieved_chunk_ids"] == [
            "gold-chunk",
            "noise-chunk",
        ]
        assert item["reciprocal_rank"] == 1.0
        assert item["ndcg_at_k"] == 1.0

    def test_candidate_recall_differs_from_final_evidence_recall(
        self,
        monkeypatch,
    ):
        dataset = {
            "version": "test-v1",
            "items": [
                {
                    "id": "q001",
                    "question": "测试问题",
                    "relevant_chunk_ids": [
                        "gold-1",
                        "gold-2",
                    ],
                    "category": "cross_chunk",
                }
            ],
        }
        candidates = [
            make_chunk("gold-1", score=0.90),
            make_chunk("gold-2", score=0.85),
            make_chunk("noise", score=0.80),
        ]

        monkeypatch.setattr(
            runner,
            "embed_text",
            lambda text: [0.0],
        )
        monkeypatch.setattr(
            runner,
            "search_chunks",
            lambda query, top_k: candidates,
        )
        monkeypatch.setattr(
            runner,
            "search_chunks_dynamic",
            lambda query, **kwargs: [candidates[0]],
        )

        result = runner.evaluate_retrieval(
            dataset,
            retrieval_strategy="dynamic",
            candidate_k=3,
            max_k=2,
            min_score=0.55,
            min_relative_score=0.85,
            metric_k=2,
        )

        item = result["items"][0]

        assert item["candidate_recall"] == 1.0
        assert item["final_evidence_recall"] == 0.5
        assert item["candidate_chunk_ids"] == [
            "gold-1",
            "gold-2",
            "noise",
        ]
        assert result["aggregate"]["candidate_recall"] == 1.0
        assert result["aggregate"]["final_evidence_recall"] == 0.5

    def test_non_rerank_strategy_records_missing_rerank_scores(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            runner,
            "embed_text",
            lambda text: [0.0],
        )
        monkeypatch.setattr(
            runner,
            "search_chunks",
            lambda query, top_k: [make_chunk()],
        )
        monkeypatch.setattr(
            runner,
            "search_chunks_dynamic",
            lambda query, **kwargs: [make_chunk()],
        )

        result = runner.evaluate_retrieval(
            make_dataset(),
            retrieval_strategy="dynamic",
            metric_k=2,
        )

        item = result["items"][0]

        assert item["retrieved_retrieval_scores"] == [0.9]
        assert item["retrieved_rerank_scores"] == [None]
        assert item["retrieved_rerank_probabilities"] == [None]

    def test_invalid_strategy_raises_error(self):
        with pytest.raises(ValueError):
            runner.evaluate_retrieval(
                make_dataset(),
                retrieval_strategy="invalid",
            )

    def test_dynamic_output_path_does_not_overwrite_fixed_result(self):
        output_path = runner.build_output_path(
            dataset_version="test-v1",
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
            "dynamic_dataset-test-v1"
            "_candidate7_max3_score060_relative080_metric2.json"
        )

    def test_fixed_output_path_contains_dataset_version(self):
        output_path = runner.build_output_path(
            dataset_version="v3",
            retrieval_strategy="fixed",
            retrieval_top_k=2,
            metric_k=5,
            neighbor_window=0,
            max_chunks=2,
            candidate_k=5,
            max_k=2,
            min_score=0.55,
            min_relative_score=0.85,
        )

        assert output_path.name == (
            "context_dataset-v3_retrieval_top2_metric5_window0_max2.json"
        )

    def test_rerank_output_path_contains_dataset_version(self):
        output_path = runner.build_output_path(
            dataset_version="v4",
            retrieval_strategy="dynamic_rerank",
            retrieval_top_k=5,
            metric_k=5,
            neighbor_window=0,
            max_chunks=5,
            candidate_k=5,
            max_k=2,
            min_score=0.55,
            min_relative_score=0.85,
            min_probability=0.50,
            min_relative_probability=0.80,
        )

        assert output_path.name == (
            "rerank_dataset-v4"
            "_candidate5_max2_prob050_relative080_metric5.json"
        )
