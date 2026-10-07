from typing import Any
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from openai import OpenAI
from config import settings
from pathlib import Path
from document_parser import parse_document
from pgvector_retriever import store_chunks, search_chunks
from chunk_builder import build_structure_sentence_chunks
from prompt_builder import (
    RAG_SYSTEM_PROMPT,
    build_rag_user_prompt,
)
from document_store import (
    calculate_sha256,
    save_document,
    list_documents,
    update_document_metadata,
    save_chunks,
    find_document_by_hash,
    add_document_alias,
)

app = FastAPI(title="企业级RAG智能知识库", version="0.2.0")


DATA_DIR = Path(__file__).resolve().parent / "data"

DOCUMENTS_DIR = DATA_DIR / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


class ChatRequest(BaseModel):
    question: str


@app.get("/", summary="服务首页，确认服务正常运行")
def root():
    return {"message": "企业级智能知识库项目已启动"}


@app.get("/health", summary="健康检查接口，监控服务存活状态")
def health():
    # 返回健康状态和服务名称
    return {"status": "ok", "service": "enterprise_rag_agent"}


@app.post("/chat", summary="调用DeekSeek大模型回答用户问题")
def chat(request: ChatRequest):
    if not settings.llm_api_key:
        return {"answer": "请先在.env文件中配置LLM_API_KEY"}

    retrieved_chunks = search_chunks(
        request.question,
        top_k=5,
    )

    if not retrieved_chunks:
        return {
            "answer": "知识库中没有足够依据回答该问题。",
            "sources": [],
        }

    user_prompt = build_rag_user_prompt(
        request.question,
        retrieved_chunks,
    )
    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )

    messages: Any = (
        [
            {
                "role": "system",
                "content": RAG_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
    )

    answer = response.choices[0].message.content or ""
    sources = [
        {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "chapter": chunk.get("chapter"),
            "heading": chunk.get("heading"),
            "score": chunk["score"],
        }
        for chunk in retrieved_chunks
    ]
    return {
        "answer": answer,
        "sources": sources,
    }


@app.post("/documents/upload", summary="上传文档到本地知识库")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        return {"status": "error", "message": "文件名无效"}

    safe_name = Path(file.filename).name
    content = await file.read()
    file_hash = calculate_sha256(content)

    existing_metadata = find_document_by_hash(
        DOCUMENTS_DIR,
        file_hash,
    )

    if existing_metadata is not None:
        metadata = add_document_alias(
            DOCUMENTS_DIR,
            existing_metadata["document_id"],
            safe_name,
        )

        return {"status": "duplicate", "document": metadata, "message": "复用已有文档"}
    metadata = save_document(
        DOCUMENTS_DIR,
        safe_name,
        content,
        file_hash,
    )
    document_path = (
        DOCUMENTS_DIR / metadata["document_id"] / metadata["stored_file_name"]
    )
    text = parse_document(document_path)

    chunks = build_structure_sentence_chunks(
        metadata["document_id"],
        text,
        chunk_size=500,
        overlap=50,
    )

    chunk_count = save_chunks(
        DOCUMENTS_DIR,
        metadata["document_id"],
        chunks,
    )

    stored_count = store_chunks(chunks)
    if stored_count != chunk_count:
        raise RuntimeError("本地chunk数量与数据库写入数量不一致")

    metadata = update_document_metadata(
        DOCUMENTS_DIR,
        metadata["document_id"],
        {
            "status": "vectorized",
            "character_count": len(text),
            "chunk_count": chunk_count,
        },
    )

    return {
        "status": "vectorized",
        "document": metadata,
    }


@app.get("/documents", summary="列出知识库中的文档列表")
def get_documents():
    return {
        "documents": list_documents(DOCUMENTS_DIR),
    }
