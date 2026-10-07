import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

METADATA_FILENAME = "metadata.json"
CHUNKS_FILENAME = "chunks.json"


def calculate_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()  # 计算SHA256哈希值并返回十六进制字符串


def save_document(
    document_dir: Path,
    original_name: str,
    content: bytes,
    file_hash: str,
) -> dict:
    document_id = uuid4().hex
    document_dir = document_dir / document_id
    document_dir.mkdir(parents=True, exist_ok=False)

    safe_name = Path(original_name).name
    suffix = Path(safe_name).suffix.lower()
    stored_file_name = f"original{suffix}"

    stored_path = document_dir / stored_file_name
    stored_path.write_bytes(content)

    metadata = {
        "document_id": document_id,
        "original_name": safe_name,
        "stored_file_name": stored_file_name,
        "extension": suffix,
        "file_hash": file_hash,
        "status": "upload",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    metadata_path = document_dir / METADATA_FILENAME
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metadata


def list_documents(documents_dir: Path) -> list[dict]:
    documents: list[dict] = []

    if not documents_dir.exists():
        return documents

    for document_dir in sorted(documents_dir.iterdir()):
        if not document_dir.is_dir():
            continue

        metadata_path = document_dir / METADATA_FILENAME
        if not metadata_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        documents.append(metadata)

    return documents


def update_document_metadata(
    documents_dir: Path,
    document_id: str,
    updates: dict,
) -> dict:
    document_dir = documents_dir / document_id
    metadata_path = document_dir / METADATA_FILENAME

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json不存在:{metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(updates)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def save_chunks(
    documents_dir: Path,
    document_id: str,
    chunks: list[dict],
) -> int:
    document_dir = documents_dir / document_id

    if not document_dir.exists():
        raise FileNotFoundError(f"文档目录不存在：{document_dir}")
    chunks_path = document_dir / CHUNKS_FILENAME
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return len(chunks)


def find_document_by_hash(
    documents_dir: Path,
    file_hash: str,
) -> dict | None:
    for metadata in list_documents(documents_dir):
        if metadata.get("file_hash") == file_hash:
            return metadata
    return None


def add_document_alias(
    documents_dir: Path,
    document_id: str,
    alias_name: str,
) -> dict:
    document_dir = documents_dir / document_id
    metadata_path = document_dir / METADATA_FILENAME

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json不存在：{metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    aliases = metadata.get("aliases", [])

    if alias_name != metadata.get("original_name") and alias_name not in aliases:
        aliases.append(alias_name)
    metadata["aliases"] = aliases

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata
