import pytest

from retrieval_policy import select_dynamic_top_k


def make_chunks(*scores):
    return [
        {
            "chunk_id": f"chunk-{index}",
            "score": score,
        }
        for index, score in enumerate(scores)
    ]


class TestSelectDynamicTopK:
    def test_empty_candidates_returns_empty_list(self):
        result = select_dynamic_top_k(
            [],
            max_k=5,
            min_score=0.55,
            min_relative_score=0.85,
        )

        assert result == []

    def test_sorts_candidates_and_limits_result_count(self):
        chunks = make_chunks(0.75, 0.90, 0.80)

        result = select_dynamic_top_k(
            chunks,
            max_k=2,
            min_score=0.55,
            min_relative_score=0.0,
        )

        assert [chunk["chunk_id"] for chunk in result] == [
            "chunk-1",
            "chunk-2",
        ]

    def test_filters_by_absolute_score(self):
        chunks = make_chunks(0.80, 0.60, 0.40)

        result = select_dynamic_top_k(
            chunks,
            max_k=5,
            min_score=0.55,
            min_relative_score=0.0,
        )

        assert [chunk["score"] for chunk in result] == [
            0.80,
            0.60,
        ]

    def test_filters_by_relative_score(self):
        chunks = make_chunks(0.80, 0.74, 0.60)

        result = select_dynamic_top_k(
            chunks,
            max_k=5,
            min_score=0.30,
            min_relative_score=0.90,
        )

        assert [chunk["score"] for chunk in result] == [
            0.80,
            0.74,
        ]

    def test_all_low_scores_returns_empty_list(self):
        chunks = make_chunks(0.30, 0.20)

        result = select_dynamic_top_k(
            chunks,
            max_k=5,
            min_score=0.55,
            min_relative_score=0.85,
        )

        assert result == []

    @pytest.mark.parametrize(
        ("max_k", "min_score", "min_relative_score"),
        [
            (0, 0.55, 0.85),
            (5, -1.1, 0.85),
            (5, 1.1, 0.85),
            (5, 0.55, -0.1),
            (5, 0.55, 1.1),
        ],
    )
    def test_invalid_parameters_raise_error(
        self,
        max_k,
        min_score,
        min_relative_score,
    ):
        with pytest.raises(ValueError):
            select_dynamic_top_k(
                make_chunks(0.80),
                max_k=max_k,
                min_score=min_score,
                min_relative_score=min_relative_score,
            )
