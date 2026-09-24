from fastapi import FastAPI,File, UploadFile
from pydantic import BaseModel
from openai import OpenAI
from config import settings
from pathlib import Path
from document_parser import parse_document
from text_processor import split_text
from document_store import (
                            calculate_sha256,
                            save_document,
                            list_documents,
                            update_document_metadata,
)
app= FastAPI(
title="企业级RAG智能知识库",
    version="0.2.0"
)


DATA_DIR = Path(__file__).resolve().parent/"data"

DOCUMENTS_DIR = DATA_DIR/"documents"
DOCUMENTS_DIR.mkdir(parents=True,exist_ok=True)

class ChatRequest(BaseModel):
    question: str

@app.get("/",summary="服务首页，确认服务正常运行")
def root():
    return{"message":"企业级智能知识库项目已启动"}


@app.get("/health",summary="健康检查接口，监控服务存活状态")
def health():
    # 返回健康状态和服务名称
    return{"status":"ok","service":"enterprise_rag_agent"}


@app.post("/chat",summary="调用DeekSeek大模型回答用户问题")
def chat(request:ChatRequest):
    if not settings.llm_api_key:
        return{"answer":"请先在.env文件中配置LLM_API_KEY"}


    client=OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
)


    response=client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role":"user",
             "content":request.question},
    ],
)


    answer = response.choices[0].message.content
    return{"answer":answer}

@app.post("/documents/upload",summary="上传文档到本地知识库")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        return{"status":"error","message":"文件名无效"}

    safe_name = Path(file.filename).name
    content=await file.read()
    file_hash=calculate_sha256(content)

    metadata=save_document(
        DOCUMENTS_DIR,
        safe_name,
        content,
        file_hash,
    )
    document_path=(
        DOCUMENTS_DIR
        /metadata["document_id"]
        /metadata["stored_file_name"]
    )
    text = parse_document(document_path)
    chunks=split_text(text)

    metadata=update_document_metadata(
        DOCUMENTS_DIR,
        metadata["document_id"],
        {
            "status":"processed",
            "character_count":len(text),
            "chunk_count":len(chunks),
        }
    )

    return{
        "status":"success",
        "document":metadata,
    }

@app.get("/documents",summary="列出知识库中的文档列表")
def get_documents():
    return{
        "documents":list_documents(DOCUMENTS_DIR),
    }
