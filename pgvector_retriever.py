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
