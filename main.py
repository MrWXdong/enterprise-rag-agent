from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from openai import OpenAI
from config import settings
from pathlib import Path
from document_parser import parse_document
from pgvector_retriever import store_chunks, search_chunks
from chunk_builder import build_structure_sentence_chunks
from document_store import (
    calculate_sha256,
    save_document,
    list_documents,
    update_document_metadata,
    save_chunks,
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

    context_blocks = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[资料{index}]\n"
            f"文档ID：{chunk['document_id']}\n"
            f"章节：{chunk.get('chapter')or"无"}\n"
            f"小标题: {chunk.get('heading')or"无"}\n"
            f"相似度：{chunk["score"]:.4f}\n"
            f"内容：{chunk['content']}"
        )
    context = "\n\n".join(context_blocks)
    prompt = f"""你是一个企业知识库问答助手。
    
    请严格遵守以下规则：
1. 只能根据提供的资料回答。
2. 不要编造资料中没有的信息。
3. 如果资料不足以回答，请明确说明无法回答。
4. 回答应简洁、准确，并尽量引用对应资料编号。
5. 不要输出无关内容。

资料：
{context}

用户问题：
{request.question}
"""

    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": "你只基于用户提供的知识库资料回答问题。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    answer = response.choices[0].message.content
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
