import pytest

from evaluation.metrics import (
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecallAtK:
    def test_partial_hit(self):
        assert (
            recall_at_k(
                ["a", "b", "c"],
                ["b", "d"],
                k=2,
            )
            == 0.5
        )

    def test_full_hit(self):
        assert (
            recall_at_k(
                ["a", "b", "c"],
                ["a", "b"],
                k=3,
            )
            == 1.0
        )

    def test_empty_relevant_ids(self):
        assert (
            recall_at_k(
                ["a", "b"],
                [],
                k=2,
            )
            == 0.0
        )


class TestPrecisionAtK:
    def test_precision(self):
        assert (
            precision_at_k(
                ["a", "b", "c", "d"],
                ["b", "d"],
                k=4,
            )
            == 0.5
        )

    def test_empty_retrieved_ids(self):
        assert (
            precision_at_k(
                [],
                ["a"],
                k=5,
            )
            == 0.0
        )


class TestReciprocalRank:
    def test_first_rank(self):
        assert (
            reciprocal_rank(
                ["a", "b"],
                ["a"],
                k=2,
            )
            == 1.0
        )

    def test_second_rank(self):
        assert (
            reciprocal_rank(
                ["a", "b"],
                ["b"],
                k=2,
            )
            == 0.5
        )

    def test_no_hit(self):
        assert (
            reciprocal_rank(
                ["a", "b"],
                ["c"],
                k=2,
            )
            == 0.0
        )


class TestNdcgAtK:
    def test_first_rank(self):
        assert (
            ndcg_at_k(
                ["a", "b"],
                ["a"],
                k=2,
            )
            == 1.0
        )

    def test_second_rank(self):
        assert ndcg_at_k(
            ["a", "b"],
            ["b"],
            k=2,
        ) == pytest.approx(0.6309297535714575)


class TestCommonValidation:
    @pytest.mark.parametrize(
        "metric",
        [
            recall_at_k,
            precision_at_k,
            ndcg_at_k,
        ],
    )
    def test_invalid_k(self, metric):
        with pytest.raises(ValueError):
            metric(["a"], ["a"], k=0)

    def test_mean(self):
        assert mean([1.0, 0.5, 0.0]) == 0.5
