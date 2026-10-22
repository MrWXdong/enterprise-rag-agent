import argparse
import json
import time
from pathlib import Path

from pgvector_retriever import search_chunks_with_context

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
    """读取评测集 JSON。"""
    if not path.exists():
        raise FileNotFoundError(f"评测集不存在：{path}")

    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_retrieval(
    dataset: dict,
    retrieval_top_k: int = 5,
    metric_k: int = 5,
    neighbor_window: int = 1,
    max_chunks: int = 15,
) -> dict:
    """评测当前 RAG 检索策略。"""
    item_results: list[dict] = []
    metric_rows: list[dict] = []

    for item in dataset["items"]:
        query = item["question"]

        start_time = time.perf_counter()
        chunks = search_chunks_with_context(
            query,
            top_k=retrieval_top_k,
            neighbor_window=neighbor_window,
            max_chunks=max_chunks,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        # search_chunks_with_context 返回原文顺序，
        # 评测检索排名时需要重新按相似度降序排列。
        ranked_chunks = sorted(
            chunks,
            key=lambda chunk: chunk["score"],
            reverse=True,
        )
        retrieved_ids = [chunk["chunk_id"] for chunk in ranked_chunks]

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
            "relevant_chunk_ids": relevant_ids,
            "context_size": len(retrieved_ids),
            "latency_ms": round(latency_ms, 2),
        }

        if should_evaluate:
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
        }

    return {
        "dataset_version": dataset.get("version"),
        "strategy": "search_chunks_with_context",
        "retrieval_top_k": retrieval_top_k,
        "metric_k": metric_k,
        "neighbor_window": neighbor_window,
        "max_chunks": max_chunks,
        "aggregate": aggregate,
        "metrics_by_category": metrics_by_category,
        "items": item_results,
    }


def build_output_path(
    retrieval_top_k: int,
    metric_k: int,
    neighbor_window: int,
    max_chunks: int,
) -> Path:
    """根据实验参数生成结果文件名，避免不同实验互相覆盖。"""
    return RESULTS_DIR / (
        "context"
        f"_retrieval_top{retrieval_top_k}"
        f"_metric{metric_k}"
        f"_window{neighbor_window}"
        f"_max{max_chunks}.json"
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval metrics.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)

    results = evaluate_retrieval(
        dataset,
        retrieval_top_k=args.retrieval_top_k,
        metric_k=args.metric_k,
        neighbor_window=args.neighbor_window,
        max_chunks=args.max_chunks,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(
        args.retrieval_top_k,
        args.metric_k,
        args.neighbor_window,
        args.max_chunks,
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
