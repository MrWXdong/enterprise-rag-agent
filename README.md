# 企业级智能知识库问答与 Agent 任务执行系统

一个面向企业私有知识库的 AI 应用平台，目标覆盖数据治理、检索与 Rerank、评测、可信问答、工具调用、文件生成和 Multi-Agent 任务编排。

项目当前处于持续开发阶段。已完成基础 RAG 闭环、动态 Top-K 和本地 Rerank 离线评测；数据中台、评测中台、在线 Agent 和 Multi-Agent 仍在演进。

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
- 动态 Top-K 候选过滤和上下文数量控制
- 低分证据可信拒答
- 本地 CrossEncoder Rerank 和全部候选重排
- Rerank 概率筛选与候选级评测审计

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

### 动态 Top-K

系统先召回 `candidate_k` 个候选，再根据当前问题的向量分数分布选择最终证据。

### Rerank

系统支持本地 `BAAI/bge-reranker-base` CrossEncoder：

- 全部候选先执行 Rerank，再执行概率筛选
- 不会在 Rerank 前用向量阈值删除候选
- 保存向量分数、Rerank logit 和 sigmoid 概率
- Rerank 路径目前仅用于离线评测，`/chat` 仍使用动态 Top-K

## 项目结构

```text
enterprise_rag_agent/
├── retrieval_policy.py      # 动态Top-K证据筛选策略
├── main.py                    # FastAPI 接口
├── config.py                  # 环境配置
├── database.py                # PostgreSQL/PgVector 表结构
├── document_parser.py         # 文档格式识别与解析
├── text_processor.py          # 字符与句子切分
├── structure_chunker.py       # 章节结构切分
├── chunk_builder.py           # 结构化 Chunk 构建
├── document_store.py          # 文件、元数据和别名管理
├── embedding_service.py       # Embedding 模型加载与向量生成
├── reranker_service.py        # 本地 Rerank 模型和概率筛选
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

RERANK_MODEL=models/bge-reranker-base
RERANK_BATCH_SIZE=16
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

测试覆盖文档解析、切分、Chunk 构建、文件与元数据管理、Embedding、内存检索、PgVector 检索、本地 Rerank 和 API 基础接口。

## 检索评测

运行固定基线检索方案：

```powershell
python -m evaluation.runner `
  --retrieval-strategy fixed `
  --retrieval-top-k 2 `
  --neighbor-window 0 `
  --max-chunks 2 `
  --metric-k 5
```

运行动态策略检索方案：

```powershell
python -m evaluation.runner `
  --retrieval-strategy dynamic `
  --candidate-k 5 `
  --max-k 2 `
  --min-score 0.55 `
  --min-relative-score 0.85 `
  --metric-k 5
```

运行 Rerank 检索方案：

```powershell
python -m evaluation.runner `
  --retrieval-strategy dynamic_rerank `
  --candidate-k 5 `
  --max-k 2 `
  --min-probability 0.60 `
  --min-relative-probability 0.85 `
  --metric-k 5
```

评测结果保存在：

```text
evaluation/results/
```

当前评测集仍处于扩充阶段。v4 共包含 22 道问题，其中 18 道参与检索指标计算，另有 2 道无答案题和 2 道歧义题。

### v4 检索评测

| 策略 | Final Recall | Precision@5 | MRR | nDCG@5 | 平均上下文数 |
|---|---:|---:|---:|---:|---:|
| 固定 K=2 | 1.0000 | 0.6111 | 0.9722 | 0.9795 | 2.0000 |
| 动态 Top-K | 0.9722 | 0.8333 | 0.9722 | 0.9580 | 1.4091 |
| Rerank 阈值 0 | 1.0000 | 0.6111 | 1.0000 | 1.0000 | 2.0000 |
| Rerank 0.60/0.85 | 1.0000 | 0.9167 | 1.0000 | 1.0000 | 1.2727 |

Rerank 后排序指标达到当前评测集的理想值，多 Gold 题没有丢失证据，两道无答案题均返回空列表。CPU 环境下 Rerank 平均延迟约 551ms，暂未接入在线问答。

## 长期规划

本节是随技术演进持续修订的项目路线，不代表已经实现的能力。项目坚持先完成可验证的底层能力，再扩展 Agent 和 Multi-Agent，不以功能数量代替系统质量。

### 目标定位

```text
数据中台：知识如何进入、治理和追踪
评测中台：系统是否真的变好
Agent 平台：如何完成复杂任务
Artifact 层：如何交付文本、表格、报告和文件
Web 工作台：用户如何操作、审批和追踪
```

### 阶段 0：当前基础

- [x] TXT 文档上传、解析、切分和本地持久化
- [x] PgVector 向量存储和 HNSW 检索
- [x] 动态 Top-K 候选数量控制
- [x] Candidate Recall 和 Final Evidence Recall 分离
- [x] 本地 CrossEncoder Rerank 和全候选重排
- [x] Rerank 概率筛选与候选级评测审计
- [x] v4 固定 K、动态 K 和 Rerank 对比评测

### 阶段 1：检索质量与稳定性

- [ ] 将评测集扩充到 50-100 道题
- [ ] 划分阈值校准集、验证集和最终测试集
- [ ] 使用留出法或交叉验证重新校准 Rerank 阈值
- [ ] 接入 BM25 稀疏检索
- [ ] 实现 BM25 + 向量混合召回
- [ ] 加入 Query Rewrite 和复杂问题拆解
- [ ] 接入在线 Rerank，并优化 Batch、ONNX、量化或 GPU 推理
- [ ] 增加证据充分性判断和歧义澄清
- [ ] 切换 BGE-M3 并重建评测向量

### 阶段 2：数据中台

- [ ] 建立文档、数据集、Chunk 和索引版本注册表
- [ ] 记录原始文件、解析结果、Chunk 和索引的数据血缘
- [ ] 增加数据质量检查和异常数据报告
- [ ] 支持增量更新、删除同步和索引重建
- [ ] 接入对象存储
- [ ] 支持 PDF、DOCX、Excel、图片和 OCR
- [ ] 增加多租户、权限和敏感数据标记
- [ ] 记录 Embedding 模型版本和索引版本

### 阶段 3：评测中台

- [ ] 建立数据集版本、标注和审核流程
- [ ] 建立实验 Run、参数、模型和结果注册表
- [ ] 保存候选级检索、Rerank 和最终证据审计
- [ ] 支持阈值网格、消融实验和置信区间
- [ ] 增加失败样本分类和错误分析
- [ ] 建立回归门槛和 CI 自动评测
- [ ] 提供指标对比、实验历史和历史报告导出
- [ ] 建设 Web 评测面板

### 阶段 4：单 Agent 工作流

- [ ] 使用 LangGraph 建立 Typed Graph State
- [ ] 定义统一 Tool Protocol 和结构化返回格式
- [ ] 实现 Planner、Retriever、Analysis 和 Reviewer 节点
- [ ] 增加工具超时、重试、幂等、Token 预算和执行追踪
- [ ] 增加 Human-in-the-loop 人工审批
- [ ] 完成首个结构化业务工作流，例如差旅报销材料审核

### 阶段 5：Multi-Agent

- [ ] 实现 Supervisor Agent 任务路由
- [ ] 增加 Retrieval Agent、Analysis Agent、Document Agent、Action Agent 和 Reviewer Agent
- [ ] 使用共享结构化状态传递任务，不直接传递无限长文本
- [ ] 支持局部失败重试、并行执行和权限隔离
- [ ] 对简单问题保留单 RAG 路径，避免无意义的多次调用
- [ ] 增加 Agent 级任务完成率、工具正确率、引用正确率、成本和延迟评测

### 阶段 6：Artifact 输出

- [ ] 统一 `result_type`、`structured_data`、`sources`、`tasks`、`artifacts` 和 `approval_required` 返回结构
- [ ] 支持 Markdown、JSON、CSV、Excel、DOCX、PDF 和 PPT
- [ ] 使用确定性渲染工具生成文件，不让大模型直接生成二进制格式
- [ ] 支持模板、图表、引用、校验、文件权限和过期时间
- [ ] 增加文件格式、内容、数值和引用正确性测试

### 阶段 7：平台与运维

- [ ] 建设 Web 任务工作台和 SSE 流式输出
- [ ] 增加 Trace、日志、Token、成本和延迟监控
- [ ] 增加 RBAC、多租户、PII、Prompt 注入防护和审计
- [ ] 完善异常处理、限流、任务队列和失败恢复
- [ ] 增加 Docker 应用服务部署和演示环境
- [ ] 补充架构图、运行截图、演示视频和技术报告

### 路线原则

- 先完成数据和评测闭环，再扩展 Agent 能力
- 简单任务使用单 RAG，复杂任务才进入 Agent 或 Multi-Agent
- Agent 负责决策和内容，确定性工具负责文件生成和业务执行
- 每个阶段必须有测试、指标、失败分析和可复现实验
- 路线随技术发展和评测结果持续修订，不把规划能力写成已完成能力

## 当前限制

- 目前主要验证 TXT 文档处理链路
- 当前 Embedding 表固定为 512 维，切换模型需要重建向量表
- 尚未接入 BM25、LangGraph Agent 和 Multi-Agent
- 数据中台和评测中台仍处于基础阶段，尚未平台化
- Rerank 目前仅用于离线评测，`/chat` 仍使用动态 Top-K
- 评测集规模仍较小，当前指标不作为最终求职数据
- 文件本地存储尚未替换为对象存储
- 动态 Top-K 和 Rerank 阈值均基于当前 v4 小规模评测集，仍需扩充数据后重新校准
- Rerank 在 CPU 环境下平均延迟约 551ms，尚未完成在线性能优化
- 宽泛问题仍可能召回证据，需要增加歧义识别和证据充分性判断

## 安全说明

- `.env`、`data/`、`.venv/`、模型文件和 IDE 配置均已排除在 Git 之外
- API Key 只存储在本地环境变量中
- 上传文件使用 `document_id` 隔离存储
- 后续会增加文件大小、MIME、权限和内容安全检查
