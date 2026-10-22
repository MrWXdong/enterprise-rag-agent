import math


def _validate_k(k: int) -> None:
    """确保 K 是正整数。"""
    if k <= 0:
        raise ValueError("k 必须大于0")


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """计算前 K 个结果召回了多少必要 gold chunks。"""
    _validate_k(k)

    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0

    retrieved_top_k = set(retrieved_ids[:k])
    hit_count = len(retrieved_top_k & relevant_set)

    return hit_count / len(relevant_set)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """计算前 K 个结果中有多少比例是相关的。"""
    _validate_k(k)

    retrieved_top_k = retrieved_ids[:k]
    if not retrieved_top_k:
        return 0.0

    relevant_set = set(relevant_ids)
    hit_count = sum(1 for chunk_id in retrieved_top_k if chunk_id in relevant_set)

    return hit_count / len(retrieved_top_k)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int | None = None,
) -> float:
    """返回第一个正确结果的倒数排名。"""
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0

    candidate_ids = retrieved_ids[:k] if k is not None else retrieved_ids

    for rank, chunk_id in enumerate(candidate_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """按二值相关性计算 nDCG@K。"""
    _validate_k(k)

    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0

    dcg = 0.0
    for rank, chunk_id in enumerate(
        retrieved_ids[:k],
        start=1,
    ):
        if chunk_id in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_count = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))

    return dcg / idcg if idcg > 0 else 0.0


def mean(values: list[float]) -> float:
    """计算算术平均值，空列表返回 0。"""
    return sum(values) / len(values) if values else 0.0
