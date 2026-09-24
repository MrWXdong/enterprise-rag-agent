from text_processor import split_text


def build_chunks(
    document_id: str,
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    raw_chunks = split_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    chunks: list[dict] = []
    total_chunks = len(raw_chunks)

    for index, content in enumerate(raw_chunks):
        chunk_id = f"{document_id}_chunk_{index:06d}"

        previous_chunk_id = None
        if index > 0:
            previous_chunk_id = f"{document_id}_chunk_{index-1:06d}"

        next_chunk_id = None
        if index < total_chunks - 1:
            next_chunk_id = f"{document_id}_chunk_{index+1:06d}"

        chunks.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "chunk_index": index,
                "content": content,
                "heading": None,
                "section_id": None,
                "previous_chunk_id": previous_chunk_id,
                "next_chunk_id": next_chunk_id,
            }
        )

    return chunks
