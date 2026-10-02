from text_processor import split_text
from structure_chunker import (
    split_into_sections,
    split_section_by_size,
)


def attach_chunk_links(
    document_id: str,
    chunks: list[dict],
) -> list[dict]:
    total_chunks = len(chunks)

    for index, chunk in enumerate(chunks):
        chunk["document_id"] = document_id
        chunk["chunk_index"] = index
        chunk["chunk_id"] = f"{document_id}_chunk_{index:06d}"

        chunk["previous_chunk_id"] = None
        if index > 0:
            chunk["previous_chunk_id"] = f"{document_id}_chunk_{index-1:06d}"

        chunk["next_chunk_id"] = None
        if index < total_chunks - 1:
            chunk["next_chunk_id"] = f"{document_id}_chunk_{index+1:06d}"
    return chunks


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

    chunks = [
        {
            "chapter": None,
            "heading": None,
            "heading_type": None,
            "content": content,
        }
        for content in raw_chunks
    ]

    return attach_chunk_links(document_id, chunks)


def build_structure_chunks(
    document_id: str, text: str, chunk_size: int = 500, overlap: int = 50
) -> list[dict]:
    sections = split_into_sections(text)

    chunks: list[dict] = []

    for section in sections:
        section_chunks = split_section_by_size(
            section,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunks.extend(section_chunks)
    return attach_chunk_links(document_id, chunks)
