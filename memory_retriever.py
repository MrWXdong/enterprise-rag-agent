from embedding_service import embed_text, embed_texts


def search_chunks(
    chunks: list[dict],
    query: str,
    top_k: int = 5,
) -> list[dict]:
    if not chunks:
        return []

    if top_k <= 0:
        raise ValueError("top_k必须大于0")

    chunk_texts = [chunk["content"] for chunk in chunks]
    chunk_vectors = embed_texts(chunk_texts)
    query_vector = embed_text(query)

    scored_chunks: list[dict] = []

    for chunk, chunk_vector in zip(chunks, chunk_vectors, strict=True):
        score = sum(
            query_value * chunk_value
            for query_value, chunk_value in zip(
                query_vector,
                chunk_vector,
            )
        )

        result = dict(chunk)
        result["score"] = score
        scored_chunks.append(result)

    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_chunks[:top_k]
