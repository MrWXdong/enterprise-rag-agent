import math
from pathlib import Path

from sentence_transformers import CrossEncoder

from config import settings
from embedding_service import get_device

_reranker: CrossEncoder | None = None


def get_reranker_model() -> CrossEncoder:
    """只从本地目录加载 Rerank 模型。"""
    global _reranker

    if _reranker is None:
        model_path = Path(settings.rerank_model)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parent / model_path

        if not model_path.exists():
            raise FileNotFoundError(f"Rerank 模型不存在：{model_path}")

        _reranker = CrossEncoder(
            str(model_path),
            device=get_device(),
            local_files_only=True,
        )

    return _reranker


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))

    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def rerank_chunks(
    query: str,
    chunks: list[dict],
    *,
    predictor=None,
    batch_size: int | None = None,
) -> list[dict]:
    """对全部候选执行 Rerank，并按重排分数降序返回。"""
    if not chunks:
        return []

    model = predictor if predictor is not None else get_reranker_model()
    pairs = [(query, chunk["content"]) for chunk in chunks]
    scores = model.predict(
        pairs,
        batch_size=batch_size or settings.rerank_batch_size,
        show_progress_bar=False,
    )

    if len(scores) != len(chunks):
        raise ValueError("Rerank 分数数量和候选数量不一致")

    reranked_chunks = []

    for chunk, score in zip(chunks, scores, strict=True):
        reranked_chunk = dict(chunk)
        reranked_chunk["retrieval_score"] = float(chunk["score"])
        reranked_chunk["rerank_score"] = float(score)
        reranked_chunk["rerank_probability"] = _sigmoid(float(score))
        reranked_chunks.append(reranked_chunk)

    return sorted(
        reranked_chunks,
        key=lambda chunk: (
            chunk["rerank_score"],
            chunk["retrieval_score"],
            chunk["chunk_id"],
        ),
        reverse=True,
    )


def select_reranked_top_k(
    chunks: list[dict],
    *,
    max_k: int,
    min_probability: float,
    min_relative_probability: float,
) -> list[dict]:
    """在 Rerank 后按概率阈值筛选最终证据。"""
    if max_k <= 0:
        raise ValueError("max_k 必须大于0")
    if not 0 <= min_probability <= 1:
        raise ValueError("min_probability 必须在0到1之间")
    if not 0 <= min_relative_probability <= 1:
        raise ValueError("min_relative_probability 必须在0到1之间")

    if not chunks:
        return []

    ranked_chunks = sorted(
        chunks,
        key=lambda chunk: chunk["rerank_probability"],
        reverse=True,
    )
    highest_probability = ranked_chunks[0]["rerank_probability"]

    if highest_probability < min_probability:
        return []

    cutoff = max(
        min_probability,
        highest_probability * min_relative_probability,
    )

    return [
        chunk
        for chunk in ranked_chunks
        if chunk["rerank_probability"] >= cutoff
    ][:max_k]
