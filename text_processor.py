from pathlib import Path
import re

SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[。！？!?；;…])"
    r"(?![”’\"』」》）】])"
    r"|"
    r"(?<=[。！？!?；;…][”’\"』」》）】])"
)


def read_text_file(file_path: Path, encoding: str = "utf-8") -> str:
    return file_path.read_text(encoding=encoding)


def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")
    if overlap < 0:
        raise ValueError("overlap不能小于0")
    if chunk_size <= overlap:
        raise ValueError("chunk_size必须大于overlap")
    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []

    for paragraph in re.split(r"\n+", text):
        parts = SENTENCE_SPLIT_PATTERN.split(paragraph)

        for part in parts:
            sentence = part.strip()
            if sentence:
                sentences.append(sentence)

    return sentences


def split_text_by_sentences(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")
    if overlap < 0:
        raise ValueError("overlap不能小于0")
    if overlap >= chunk_size:
        raise ValueError("overlap必须小于chunk_size")

    sentences = split_sentences(text)

    if not sentences:
        return []
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if sentence_length > chunk_size:
            if current_sentences:
                chunks.append("".join(current_sentences))
                current_sentences = []
                current_length = 0

            chunks.extend(
                split_text(
                    sentence,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )
            continue
        if current_sentences and current_length + sentence_length > chunk_size:
            chunks.append("".join(current_sentences))
            overlap_sentences: list[str] = []
            overlap_length = 0
            for previous in reversed(current_sentences):
                if overlap_length + len(previous) > overlap:
                    break
                overlap_sentences.insert(0, previous)
                overlap_length += len(previous)

            current_sentences = overlap_sentences
            current_length = overlap_length

        current_sentences.append(sentence)
        current_length += sentence_length

    if current_sentences:
        chunks.append("".join(current_sentences))
    return chunks
