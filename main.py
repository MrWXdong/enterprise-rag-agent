from fastapi import FastAPI
app= FastAPI(
    title="企业级RAG智能知识库",
    version="0.1.0"
)
@app.get("/",summary="服务首页，确认服务正常运行")
def root():
    return{"message":"企业级智能知识库项目已启动"}
@app.get("/health",summary="健康检查接口，监控服务存活状态")
def health():
    # 返回健康状态和服务名称
    return{"status":"ok","service":"enterprise_rag_agent"}