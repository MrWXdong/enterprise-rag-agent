def select_dynamic_top_k(
    chunks: list[dict],
    max_k: int,
    min_score: float,
    min_relative_score: float,
) -> list[dict]:
    """按绝对分数和相对最高分比例筛选最终召回结果"""
    if max_k <= 0:
        raise ValueError("max_k 必须大于0")
    if not -1 <= min_score <= 1:
        raise ValueError("min_score 必须在[-1,1]之间")
    if not 0 <= min_relative_score <= 1:
        raise ValueError("min_relative_score 必须在[0,1]之间")

    if not chunks:
        return []
    # 不依赖调用方提前排序，函数内部重新按相似度降序排列。
    ranked_chunks = sorted(
        chunks,
        key=lambda chunk: chunk["score"],
        reverse=True,
    )
    highest_score = ranked_chunks[0]["score"]

    if highest_score < min_score:
        return []

    cutoff = max(
        min_score,
        highest_score * min_relative_score,
    )
    selected = [chunk for chunk in ranked_chunks if chunk["score"] >= cutoff]

    return selected[:max_k]


if __name__ == "__main__":
    chunks = [
        {"chunk_id": "a", "score": 0.90},
        {"chunk_id": "b", "score": 0.82},
        {"chunk_id": "c", "score": 0.50},
    ]

    result = select_dynamic_top_k(
        chunks,
        max_k=2,
        min_score=0.55,
        min_relative_score=0.85,
    )

    print("动态筛选结果：")
    for chunk in result:
        print(f"chunk_id={chunk['chunk_id']}, " f"score={chunk['score']}")

    print("\n动态阈值：")
    print(0.90 * 0.85)
