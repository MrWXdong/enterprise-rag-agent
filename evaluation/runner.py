import argparse
import json
import time
from pathlib import Path

from embedding_service import embed_text
from pgvector_retriever import (
    search_chunks_dynamic,
    search_chunks_with_context,
)

from evaluation.metrics import (
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_DIR / "evaluation" / "dataset.json"
RESULTS_DIR = PROJECT_DIR / "evaluation" / "results"


def load_dataset(path: Path) -> dict:
    """读取固定评测集。"""
    if not path.exists():
        raise FileNotFoundError(f"评测集不存在：{path}")

    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_retrieval(
    dataset: dict,
    retrieval_strategy: str = "fixed",
    retrieval_top_k: int = 5,
    metric_k: int = 5,
    neighbor_window: int = 1,
    max_chunks: int = 15,
    candidate_k: int = 5,
    max_k: int = 2,
    min_score: float = 0.55,
    min_relative_score: float = 0.85,
) -> dict:
    """运行检索评测，并汇总整体及分类指标。"""

    if retrieval_strategy not in {"fixed", "dynamic"}:
        raise ValueError("retrieval_strategy 必须是 fixed 或 dynamic")

    if retrieval_strategy == "dynamic":
        if candidate_k <= 0:
            raise ValueError("candidate_k 必须大于0")
        if max_k <= 0:
            raise ValueError("max_k 必须大于0")
        if not -1 <= min_score <= 1:
            raise ValueError("min_score 必须在 -1 到 1 之间")
        if not 0 <= min_relative_score <= 1:
            raise ValueError("min_relative_score 必须在 0 到 1 之间")
    else:
        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k 必须大于0")
        if neighbor_window < 0:
            raise ValueError("neighbor_window 不能小于0")
        if max_chunks <= 0:
            raise ValueError("max_chunks 必须大于0")

    # 预热模型，避免首次加载时间污染单题延迟。
    embed_text("模型预热")

    item_results: list[dict] = []
    metric_rows: list[dict] = []

    for item in dataset["items"]:
        query = item["question"]

        start_time = time.perf_counter()

        if retrieval_strategy == "dynamic":
            chunks = search_chunks_dynamic(
                query,
                candidate_k=candidate_k,
                max_k=max_k,
                min_score=min_score,
                min_relative_score=min_relative_score,
            )
        else:
            chunks = search_chunks_with_context(
                query,
                top_k=retrieval_top_k,
                neighbor_window=neighbor_window,
                max_chunks=max_chunks,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # 检索评测统一按相似度降序计算排名。
        ranked_chunks = sorted(
            chunks,
            key=lambda chunk: chunk["score"],
            reverse=True,
        )

        retrieved_ids = [chunk["chunk_id"] for chunk in ranked_chunks]
        retrieved_scores = [round(float(chunk["score"]), 6) for chunk in ranked_chunks]
        retrieved_chunk_indices = [chunk["chunk_index"] for chunk in ranked_chunks]
        retrieved_headings = [chunk.get("heading") for chunk in ranked_chunks]

        relevant_ids = item.get("relevant_chunk_ids", [])
        category = item.get("category", "unknown")
        should_evaluate = category not in {"no_answer", "ambiguous"} and bool(
            relevant_ids
        )

        item_result = {
            "id": item["id"],
            "question": query,
            "category": category,
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_scores": retrieved_scores,
            "retrieved_chunk_indices": retrieved_chunk_indices,
            "retrieved_headings": retrieved_headings,
            "relevant_chunk_ids": relevant_ids,
            "context_size": len(retrieved_ids),
            "latency_ms": round(latency_ms, 2),
        }

        if should_evaluate:
            # 上下文指标使用实际返回数量，以观察动态 K 的噪声变化。
            context_k = max(len(retrieved_ids), 1)

            item_result.update(
                {
                    "recall_at_k": recall_at_k(
                        retrieved_ids,
                        relevant_ids,
                        k=metric_k,
                    ),
                    "precision_at_k": precision_at_k(
                        retrieved_ids,
                        relevant_ids,
                        k=metric_k,
                    ),
                    "reciprocal_rank": reciprocal_rank(
                        retrieved_ids,
                        relevant_ids,
                        k=metric_k,
                    ),
                    "ndcg_at_k": ndcg_at_k(
                        retrieved_ids,
                        relevant_ids,
                        k=metric_k,
                    ),
                    "context_recall": recall_at_k(
                        retrieved_ids,
                        relevant_ids,
                        k=context_k,
                    ),
                    "context_precision": precision_at_k(
                        retrieved_ids,
                        relevant_ids,
                        k=context_k,
                    ),
                }
            )

            metric_rows.append(item_result)

        item_results.append(item_result)

    aggregate = {
        "evaluated_item_count": len(metric_rows),
        "recall_at_k": mean([row["recall_at_k"] for row in metric_rows]),
        "precision_at_k": mean([row["precision_at_k"] for row in metric_rows]),
        "mrr": mean([row["reciprocal_rank"] for row in metric_rows]),
        "ndcg_at_k": mean([row["ndcg_at_k"] for row in metric_rows]),
        "context_recall": mean([row["context_recall"] for row in metric_rows]),
        "context_precision": mean([row["context_precision"] for row in metric_rows]),
        "average_context_size": mean([row["context_size"] for row in item_results]),
        "average_latency_ms": mean([row["latency_ms"] for row in item_results]),
    }

    category_names = sorted({row["category"] for row in metric_rows})
    metrics_by_category: dict[str, dict] = {}

    for category in category_names:
        category_rows = [row for row in metric_rows if row["category"] == category]

        metrics_by_category[category] = {
            "item_count": len(category_rows),
            "recall_at_k": mean([row["recall_at_k"] for row in category_rows]),
            "precision_at_k": mean([row["precision_at_k"] for row in category_rows]),
            "mrr": mean([row["reciprocal_rank"] for row in category_rows]),
            "ndcg_at_k": mean([row["ndcg_at_k"] for row in category_rows]),
            "context_recall": mean([row["context_recall"] for row in category_rows]),
            "context_precision": mean(
                [row["context_precision"] for row in category_rows]
            ),
        }

    return {
        "dataset_version": dataset.get("version"),
        "strategy": retrieval_strategy,
        "retrieval_top_k": retrieval_top_k,
        "candidate_k": candidate_k,
        "max_k": max_k,
        "min_score": min_score,
        "min_relative_score": min_relative_score,
        "metric_k": metric_k,
        "neighbor_window": neighbor_window,
        "max_chunks": max_chunks,
        "aggregate": aggregate,
        "metrics_by_category": metrics_by_category,
        "items": item_results,
    }


def build_output_path(
    retrieval_strategy: str,
    retrieval_top_k: int,
    metric_k: int,
    neighbor_window: int,
    max_chunks: int,
    candidate_k: int,
    max_k: int,
    min_score: float,
    min_relative_score: float,
) -> Path:
    """根据实验参数生成独立结果文件名。"""

    if retrieval_strategy == "dynamic":
        score_name = f"{min_score:.2f}".replace(".", "")
        relative_name = f"{min_relative_score:.2f}".replace(".", "")

        return RESULTS_DIR / (
            "dynamic"
            f"_candidate{candidate_k}"
            f"_max{max_k}"
            f"_score{score_name}"
            f"_relative{relative_name}"
            f"_metric{metric_k}.json"
        )

    return RESULTS_DIR / (
        "context"
        f"_retrieval_top{retrieval_top_k}"
        f"_metric{metric_k}"
        f"_window{neighbor_window}"
        f"_max{max_chunks}.json"
    )


def parse_args() -> argparse.Namespace:
    """解析评测命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval metrics.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--retrieval-strategy",
        choices=["fixed", "dynamic"],
        default="fixed",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--metric-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--neighbor-window",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--min-relative-score",
        type=float,
        default=0.85,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)

    results = evaluate_retrieval(
        dataset,
        retrieval_strategy=args.retrieval_strategy,
        retrieval_top_k=args.retrieval_top_k,
        metric_k=args.metric_k,
        neighbor_window=args.neighbor_window,
        max_chunks=args.max_chunks,
        candidate_k=args.candidate_k,
        max_k=args.max_k,
        min_score=args.min_score,
        min_relative_score=args.min_relative_score,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = build_output_path(
        args.retrieval_strategy,
        args.retrieval_top_k,
        args.metric_k,
        args.neighbor_window,
        args.max_chunks,
        args.candidate_k,
        args.max_k,
        args.min_score,
        args.min_relative_score,
    )

    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            results["aggregate"],
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"结果已保存到：{output_path}")


if __name__ == "__main__":
    main()
