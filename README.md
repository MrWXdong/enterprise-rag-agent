# 企业级智能知识库问答与 Agent 任务执行系统

一个面向企业私有知识库的 RAG 应用项目，目标覆盖文档处理、向量检索、可信问答和后续 Agent 任务编排。

项目当前处于持续开发阶段。已完成基础 RAG 闭环，Agent、混合检索和 Rerank 等能力在后续迭代中实现。

## 项目目标

- 支持企业多格式文档的解析、清洗、结构化切分和持久化
- 使用 PgVector 构建可扩展的向量检索能力
- 结合 DeepSeek 完成基于私有知识的 RAG 问答
- 返回答案来源，降低无依据回答和知识库外编造
- 通过固定评测集比较不同切分和检索策略
- 后续使用 LangGraph 实现任务拆解、工具调用和多步执行

## 已完成能力

- FastAPI 接口服务
- 文档上传、列表和本地文件管理
- TXT 文档解析和格式分派
- 固定字符切分基线
- 基于章节标题的结构感知切分
- 中文句子感知切分
- 结构化 Chunk 和相邻 Chunk 关系
- SHA-256 文件内容去重与别名管理
- BGE 中文 Embedding
- PostgreSQL + PgVector 向量持久化
- HNSW 向量索引
- Top-K 向量检索
- 相邻上下文扩展和结果去重
- DeepSeek RAG 问答和来源返回
- Pytest 自动化测试
- Recall@K、Precision@K、MRR、nDCG@K 评测基础设施

## 系统架构

```mermaid
flowchart LR
    A[文档上传] --> B[格式识别与解析]
    B --> C[结构感知切分]
    C --> D[句子感知切分]
    D --> E[BGE Embedding]
    E --> F[(PostgreSQL + PgVector)]

    G[用户问题] --> H[Query Embedding]
    H --> I[Top-K 向量召回]
    I --> J[相邻上下文扩展]
    J --> K[Prompt 构建]
    K --> L[DeepSeek]
    L --> M[答案 + 来源引用]
```

## 技术栈

| 模块 | 技术 |
| --- | --- |
| API | FastAPI、Uvicorn、Pydantic |
| LLM | DeepSeek OpenAI 兼容接口 |
| Embedding | Sentence Transformers、BAAI/bge-small-zh-v1.5 |
| 向量存储 | PostgreSQL、PgVector、HNSW |
| 文档处理 | TXT 解析、章节识别、句子边界切分 |
| 测试与评测 | Pytest、Recall@K、Precision@K、MRR、nDCG@K |
| 部署 | Docker Compose |

## 核心设计

### 多策略切分

项目保留三种切分方案，用于对比和后续评测：

1. `build_chunks()`：固定字符切分基线
2. `build_structure_chunks()`：章节结构优先，字符切分兜底
3. `build_structure_sentence_chunks()`：章节结构 + 句子边界优先，字符切分兜底

结构化 Chunk 会保留：

```text
chunk_id
document_id
chunk_index
content
chapter
heading
heading_type
previous_chunk_id
next_chunk_id
```

### 文件级去重

上传文件时计算 SHA-256：

- 相同内容不会重复保存和重复生成向量
- 新文件名追加到已有文档的 `aliases`
- 不同内容即使文件名相同，也会生成新的 `document_id`

### 关联上下文扩展

基础检索只返回相似度最高的种子 Chunk。系统可以继续查询同文档中的前后 Chunk，并按照 `chunk_id` 去重，减少因切分造成的上下文缺失。

## 项目结构

```text
enterprise_rag_agent/
├── main.py                    # FastAPI 接口
├── config.py                  # 环境配置
├── database.py                # PostgreSQL/PgVector 表结构
├── document_parser.py         # 文档格式识别与解析
├── text_processor.py          # 字符与句子切分
├── structure_chunker.py       # 章节结构切分
├── chunk_builder.py           # 结构化 Chunk 构建
├── document_store.py          # 文件、元数据和别名管理
├── embedding_service.py       # Embedding 模型加载与向量生成
├── pgvector_retriever.py      # 向量写入、检索和上下文扩展
├── memory_retriever.py        # 内存检索基线
├── prompt_builder.py          # RAG Prompt 构建
├── evaluation/
│   ├── dataset.json           # 固定评测集
│   ├── metrics.py             # 检索指标
│   └── runner.py              # 评测运行器
├── docker-compose.yml         # PgVector 数据库服务
├── requirements.txt
├── .env.example
└── test_*.py
```

## 快速启动

### 1. 创建虚拟环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 配置环境变量

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 中配置：

```text
LLM_API_KEY=your_deepseek_api_key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash

EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

POSTGRES_USER=rag_user
POSTGRES_PASSWORD=change_me
POSTGRES_DB=rag_db
POSTGRES_PORT=5432

DATABASE_URL=postgresql://rag_user:change_me@localhost:5432/rag_db
```

`.env` 不应提交到 Git。

### 3. 启动数据库

```powershell
docker compose up -d
```

初始化 PgVector、数据表和 HNSW 索引：

```powershell
python -c "from database import init_database; init_database()"
```

### 4. 启动服务

首次下载 Embedding 模型后，可以使用离线模式：

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
uvicorn main:app --reload --port 8000
```

访问接口文档：

```text
http://127.0.0.1:8000/docs
```

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 服务首页 |
| GET | `/health` | 健康检查 |
| POST | `/documents/upload` | 上传并处理文档 |
| GET | `/documents` | 查看文档列表 |
| POST | `/chat` | 基于知识库回答问题 |

### 问答示例

```json
{
  "question": "一类城市住宿费每天最高多少？"
}
```

返回：

```json
{
  "answer": "一类城市普通员工每晚最高报销600元。",
  "sources": [
    {
      "chunk_id": "document_id_chunk_000004",
      "document_id": "document_id",
      "chapter": "第一章 公司差旅与报销管理制度",
      "heading": "第四条 住宿费用标准",
      "score": 0.82
    }
  ]
}
```

## 测试

```powershell
pytest -q
```

测试覆盖文档解析、切分、Chunk 构建、文件与元数据管理、Embedding、内存检索、PgVector 检索和 API 基础接口。

## 检索评测

运行关联上下文检索方案：

```powershell
python -m evaluation.runner `
  --retrieval-top-k 5 `
  --metric-k 5 `
  --neighbor-window 1 `
  --max-chunks 15
```

运行基础检索方案：

```powershell
python -m evaluation.runner `
  --retrieval-top-k 5 `
  --metric-k 5 `
  --neighbor-window 0 `
  --max-chunks 5
```

评测结果保存在：

```text
evaluation/results/
```

当前评测集仍处于扩充阶段。后续会使用固定问题集对比不同切分、混合检索和 Rerank 方案。

## 后续路线

- [ ] 扩充检索评测集和测试文档
- [ ] 接入 BM25 稀疏检索
- [ ] 实现 BM25 + 向量混合召回
- [ ] 接入 Rerank 精排
- [ ] 增加动态 Top-K 和上下文预算
- [ ] 增加证据充分性判断和可信拒答
- [ ] 增加 PDF 与 DOCX 解析
- [ ] 使用 LangGraph 实现 Agent 工具编排
- [ ] 完善日志、异常处理、限流和监控
- [ ] 增加 Docker 应用服务部署
- [ ] 补充架构图、运行截图和演示视频
- [ ] 切换 BGE-M3 并重建评测向量

## 当前限制

- 目前主要验证 TXT 文档处理链路
- 当前 Embedding 表固定为 512 维，切换模型需要重建向量表
- 尚未接入 BM25、Rerank 和 LangGraph Agent
- 评测集规模仍较小，当前指标不作为最终求职数据
- 文件本地存储尚未替换为对象存储

## 安全说明

- `.env`、`data/`、`.venv/`、模型文件和 IDE 配置均已排除在 Git 之外
- API Key 只存储在本地环境变量中
- 上传文件使用 `document_id` 隔离存储
- 后续会增加文件大小、MIME、权限和内容安全检查
