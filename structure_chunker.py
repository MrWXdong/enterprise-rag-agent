import re
from text_processor import split_text

CHAPTER_PATTERN = re.compile(r"^第[一二三四五六七八九十百零〇万千两\d]+章(?:\s|$)")
ARTICLE_PATTERN = re.compile(r"^第[一二三四五六七八九十百零〇万千两\d]+条(?:\s|$)")
QUESTION_PATTERN = re.compile(r"^问题[一二三四五六七八九十百零〇万千两\d]+[:：]")
MARKDOWN_PATTERN = re.compile(r"^#{1,6}\s+")

HEADING_PATTERNS = [
    CHAPTER_PATTERN,
    ARTICLE_PATTERN,
    QUESTION_PATTERN,
    MARKDOWN_PATTERN,
]


def is_heading(line: str) -> bool:
    normalized_line = line.strip()

    return any(pattern.match(normalized_line) for pattern in HEADING_PATTERNS)


def flush_section(
    sections: list[dict],
    chapter: str | None,
    heading: str | None,
    heading_type: str | None,
    lines: list[str],
) -> None:
    body = "\n".join(lines).strip()

    if not body:
        return

    sections.append(
        {
            "chapter": chapter,
            "heading": heading,
            "heading_type": heading_type,
            "body": body,
        }
    )


def split_into_sections(text: str) -> list[dict]:
    sections: list[dict] = []
    current_chapter: str | None = None
    current_heading: str | None = None
    current_heading_type: str | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if line and is_heading(line):
            flush_section(
                sections,
                current_chapter,
                current_heading,
                current_heading_type,
                current_lines,
            )

            heading_type = detect_heading_type(line)

            if CHAPTER_PATTERN.match(line):
                current_chapter = line
                current_heading = None
                current_heading_type = None
            else:
                current_heading = line
                current_heading_type = heading_type

            current_lines = []
            continue

        current_lines.append(raw_line)

    flush_section(
        sections,
        current_chapter,
        current_heading,
        current_heading_type,
        current_lines,
    )
    return sections


def detect_heading_type(line: str) -> str | None:
    normalized_line = line.strip()

    if CHAPTER_PATTERN.match(normalized_line):
        return "chapter"

    if ARTICLE_PATTERN.match(normalized_line):
        return "article"

    if QUESTION_PATTERN.match(normalized_line):
        return "question"

    if MARKDOWN_PATTERN.match(normalized_line):
        return "markdown"

    return None


def split_section_by_size(
    section: dict,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    chapter = section.get("chapter")
    heading = section.get("heading")
    heading_type = section.get("heading_type")
    body = section.get("body", "").strip()

    if not body:
        return []

    title = heading or chapter or ""
    title_prefix = f"{title}\n" if title else ""

    if chunk_size <= len(title_prefix):
        raise ValueError("chunk_size 必须要大于标题长度")

    body_chunk_size = chunk_size - len(title_prefix)

    if overlap >= body_chunk_size:
        raise ValueError("overlap 必须要小于可用正文的分块长度")

    body_chunks = split_text(
        body,
        chunk_size=body_chunk_size,
        overlap=overlap,
    )

    chunks: list[dict] = []
    for index, body_chunk in enumerate(body_chunks):
        content = f"{title_prefix}{body_chunk}".strip()

        chunks.append(
            {
                "chapter": chapter,
                "heading": heading,
                "heading_type": heading_type,
                "chunk_index_in_section": index,
                "content": content,
            }
        )
    return chunks
