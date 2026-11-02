import pytest

from reranker_service import rerank_chunks, select_reranked_top_k


class FakePredictor:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None
        self.batch_size = None

    def predict(self, pairs, *, batch_size, show_progress_bar):
        self.pairs = pairs
        self.batch_size = batch_size
        assert show_progress_bar is False
        return self.scores


def make_chunk(chunk_id, score):
    return {
        "chunk_id": chunk_id,
        "content": f"内容-{chunk_id}",
        "score": score,
    }


class TestRerankChunks:
    def test_empty_chunks_returns_empty_list(self):
        assert rerank_chunks("问题", []) == []

    def test_reranks_and_preserves_retrieval_scores(self):
        chunks = [
            make_chunk("chunk-1", 0.90),
            make_chunk("chunk-2", 0.80),
        ]
        predictor = FakePredictor([0.0, 2.0])

        results = rerank_chunks(
            "测试问题",
            chunks,
            predictor=predictor,
        )

        assert predictor.pairs == [
            ("测试问题", "内容-chunk-1"),
            ("测试问题", "内容-chunk-2"),
        ]
        assert [chunk["chunk_id"] for chunk in results] == [
            "chunk-2",
            "chunk-1",
        ]
        assert results[0]["retrieval_score"] == 0.80
        assert results[0]["rerank_score"] == 2.0
        assert results[0]["rerank_probability"] == pytest.approx(
            0.880797,
            abs=0.000001,
        )
        assert "rerank_score" not in chunks[0]

    def test_score_count_mismatch_raises_error(self):
        predictor = FakePredictor([0.90])

        with pytest.raises(ValueError):
            rerank_chunks(
                "测试问题",
                [
                    make_chunk("chunk-1", 0.90),
                    make_chunk("chunk-2", 0.80),
                ],
                predictor=predictor,
            )


class TestSelectRerankedTopK:
    def make_reranked_chunks(self):
        return [
            {
                "chunk_id": "chunk-1",
                "rerank_probability": 0.90,
            },
            {
                "chunk_id": "chunk-2",
                "rerank_probability": 0.80,
            },
            {
                "chunk_id": "chunk-3",
                "rerank_probability": 0.40,
            },
        ]

    def test_filters_and_limits_after_rerank(self):
        results = select_reranked_top_k(
            self.make_reranked_chunks(),
            max_k=2,
            min_probability=0.50,
            min_relative_probability=0.80,
        )

        assert [chunk["chunk_id"] for chunk in results] == [
            "chunk-1",
            "chunk-2",
        ]

    def test_all_low_probabilities_return_empty_list(self):
        assert (
            select_reranked_top_k(
                self.make_reranked_chunks(),
                max_k=2,
                min_probability=0.95,
                min_relative_probability=0.80,
            )
            == []
        )

    @pytest.mark.parametrize(
        ("max_k", "min_probability", "min_relative_probability"),
        [
            (0, 0.50, 0.80),
            (2, -0.1, 0.80),
            (2, 1.1, 0.80),
            (2, 0.50, -0.1),
            (2, 0.50, 1.1),
        ],
    )
    def test_invalid_parameters_raise_error(
        self,
        max_k,
        min_probability,
        min_relative_probability,
    ):
        with pytest.raises(ValueError):
            select_reranked_top_k(
                self.make_reranked_chunks(),
                max_k=max_k,
                min_probability=min_probability,
                min_relative_probability=min_relative_probability,
            )
