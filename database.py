import psycopg
from pgvector.psycopg import register_vector

from config import settings

VECTOR_DIMENSION = 512


def connect_database() -> psycopg.Connection:
    return psycopg.connect(settings.database_url)


def get_connection() -> psycopg.Connection:
    connection = connect_database()
    register_vector(connection)
    return connection


def init_database() -> None:
    connection = connect_database()

    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    chapter TEXT,
                    heading TEXT,
                    heading_type TEXT,
                    embedding_model TEXT NOT NULL,
                    embedding vector(512) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (document_id, chunk_index)
                )
                """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS
                    document_chunks_document_id_idx
                ON document_chunks (document_id)
                """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS
                    document_chunks_embedding_hnsw_idx
                ON document_chunks
                USING hnsw (embedding vector_cosine_ops)
                """)

        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
