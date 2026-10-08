from pgvector import Vector
from config import settings
from database import get_connection
from embedding_service import embed_text, embed_texts


def store_chunks(chunks: list[dict]) -> int:
    if not chunks:
        return 0

    document_id = chunks[0]["document_id"]
    contents = [chunk["content"] for chunk in chunks]
    vectors = embed_texts(contents)

    if len(vectors) != len(chunks):
        raise ValueError("chunk数量和向量数量不一致")
    rows = []

    for chunk, vector in zip(
        chunks,
        vectors,
        strict=True,
    ):
        rows.append(
            (
                chunk["chunk_id"],
                chunk["document_id"],
                chunk["chunk_index"],
                chunk["content"],
                chunk.get("chapter"),
                chunk.get("heading"),
                chunk.get("heading_type"),
                settings.embedding_model,
                Vector(vector),
            )
        )
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM document_chunks
                WHERE document_id = %s
                """,
                (document_id,),
            )

            cursor.executemany(
                """
                INSERT INTO document_chunks(
                chunk_id,
                document_id,
                chunk_index,
                content,
                chapter,
                heading,
                heading_type,
                embedding_model,
                embedding
            )
            VALUES(
                %s,%s,%s,%s,%s,
                %s,%s,%s,%s
                )
            ON CONFLICT(chunk_id)
            DO UPDATE SET 
                content = EXCLUDED.content,
                chapter = EXCLUDED.chapter,
                heading = EXCLUDED.heading,
                heading_type = EXCLUDED.heading_type,
                embedding_model = EXCLUDED.embedding_model,
                embedding = EXCLUDED.embedding
            """,
                rows,
            )
    return len(chunks)


def search_chunks(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k 必须大于0")

    query_vector = Vector(embed_text(query))

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    chunk_id,
                    document_id,
                    chunk_index,
                    content,
                    chapter,
                    heading,
                    heading_type,
                    embedding_model,
                    1-(embedding <=> %s::vector) AS score
                FROM document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    query_vector,
                    query_vector,
                    top_k,
                ),
            )
            rows = cursor.fetchall()
    return [
        {
            "chunk_id": row[0],
            "document_id": row[1],
            "chunk_index": row[2],
            "content": row[3],
            "chapter": row[4],
            "heading": row[5],
            "heading_type": row[6],
            "embedding_model": row[7],
            "score": float(row[8]),
        }
        for row in rows
    ]


def row_to_chunk(row: tuple) -> dict:
    return {
        "chunk_id": row[0],
        "document_id": row[1],
        "chunk_index": row[2],
        "content": row[3],
        "chapter": row[4],
        "heading": row[5],
        "heading_type": row[6],
        "embedding_model": row[7],
        "score": float(row[8]),
    }


def search_chunks_with_context(
    query: str,
    top_k: int = 5,
    neighbor_window: int = 1,
    max_chunks: int = 15,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k 必须大于0")
    if neighbor_window < 0:
        raise ValueError("neighbor_window 不能小于0")
    if max_chunks <= 0:
        raise ValueError("max_chunks 必须大于0")

    seed_chunks = search_chunks(
        query,
        top_k=top_k,
    )
    if not seed_chunks:
        return []
    query_vector = Vector(embed_text(query))
    expanded: dict[str, dict] = {}

    with get_connection() as connection:
        with connection.cursor() as cursor:
            for seed in seed_chunks:
                min_index = max(
                    0,
                    seed["chunk_index"] - neighbor_window,
                )
                max_index = seed["chunk_index"] + neighbor_window
                cursor.execute(
                    """
                    SELECT
                        chunk_id,
                        document_id,
                        chunk_index,
                        content,
                        chapter,
                        heading,
                        heading_type,
                        embedding_model,
                        1-(embedding<=>%s::vector)AS score
                    FROM document_chunks
                    WHERE document_id =%s
                      AND chunk_index BETWEEN %s AND %s
                    
                """,
                    (
                        query_vector,
                        seed["document_id"],
                        min_index,
                        max_index,
                    ),
                )

                for row in cursor.fetchall():
                    chunk = row_to_chunk(row)
                    existing = expanded.get(chunk["chunk_id"])

                    if existing is None or chunk["score"] > existing["score"]:
                        expanded[chunk["chunk_id"]] = chunk

    selected = sorted(
        expanded.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:max_chunks]

    return sorted(
        selected,
        key=lambda item: (
            item["document_id"],
            item["chunk_index"],
        ),
    )
