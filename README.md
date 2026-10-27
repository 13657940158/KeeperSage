# KeeperSage 企业文档系统

基于 RAG 的企业级文档知识库：上传 PDF/MD 文档，自动解析、分块、向量化入库，并通过 Web 页面进行多路召回的智能问答，支持流式输出与原图引用。

> **全链路独立完成**：文档解析 → 混合检索 → 多路召回融合 → 重排 → 流式生成，覆盖 RAG 应用从数据摄入到答案交付的完整工程链路。

---

## 项目亮点

- **完整 RAG 双流水线**：导入（解析/分块/向量化/入库）与查询（改写/检索/融合/重排/生成）均基于 **LangGraph** 状态机实现，节点化编排、流式推进、可观测。
- **混合检索**：BGE-M3 同时产出 dense（语义）与 sparse（学习型稀疏）向量，`WeightedRanker` 加权融合，兼顾语义泛化与关键词精确匹配。
- **三路并行召回**：普通向量检索 / **HyDE** 假设性文档检索 / **MCP 协议联网搜索**（阿里云百炼），多源互补提升召回率。
- **RRF + Cross-Encoder 二级排序**：先用 Reciprocal Rank Fusion 按排名融合多路结果，再用重排模型精排，兼顾效率与精度。
- **多模态文档解析**：接入 MinerU 将 PDF 转为 Markdown，图片经多模态 LLM 生成描述后随分块入库，问答时可回显原图。
- **流式问答体验**：SSE 实时推送节点进度与增量答案，前端展示"阶段进度"，答案自动关联参考图片。

## 系统架构

```
导入流水线（LangGraph）
PDF/MD → MinerU解析 → 图片多模态描述 → 分块 → 向量化(dense+sparse)
      → Milvus入库 → MinIO原件持久化 → MongoDB记录

查询流水线（LangGraph）
用户提问 → 查询改写 + 设备名抽取确认
        → ┌ 普通向量检索（dense+sparse 混合）
          ├ HyDE 假设性文档检索
          └ MCP 联网搜索
        → RRF 排名融合 → Cross-Encoder 重排 → LLM 生成（SSE 流式）→ 答案+图片
```

## 功能特性

- 批量上传 PDF / MD，按「日期 / 任务ID」分层存储，实时轮询处理进度与节点日志
- 解析产物（Markdown、图片）自动结构化，图片上传对象存储并回链
- 基于会话的多轮问答，历史记录持久化（MongoDB）
- 答案自动携带参考图片，前端 Markdown 化展示
- 双页面 Web 界面（导入 / 问答），顶部导航一键切换

## 技术栈

| 层 | 技术 |
|---|---|
| 编排 | LangGraph（状态机）、FastAPI、BackgroundTasks |
| 解析 | MinerU（本地 API）、多模态 LLM 图片描述 |
| 向量 | BGE-M3（dense + sparse）、Milvus 混合检索 |
| 排序 | RRF、Cross-Encoder 重排（DashScope） |
| 生成 | OpenAI 兼容 LLM、SSE 流式输出 |
| 联网 | MCP 协议（Streamable HTTP）接入百炼搜索 |
| 存储 | Milvus / MongoDB / MinIO |
| 前端 | 原生 HTML/JS（无框架，零构建依赖） |

## 快速开始

### 环境要求

- Python 3.11
- NVIDIA GPU（建议 8GB+ 显存，运行 BGE-M3；无 GPU 可用 CPU 但较慢）
- Docker：Milvus、MongoDB、MinIO
- MinerU 本地解析服务（或自行改接云端解析 API）

### 部署步骤

```bash
# 1. 拉取代码、安装依赖
git clone <本仓库地址> && cd KeeperSage
pip install -r requirements.txt

# 注意：mcp 库须为 1.x（pip install "mcp<2"），2.x 与百炼 MCP 服务协议不兼容

# 2. 启动中间件（Docker）
docker run -d -p 27017:27017 --name mongo mongo:6
docker run -d -p 9000:9000 -p 9001:9001 --name minio minio/minio server /data --console-address ":9001"
# Milvus 参考官方 standalone docker-compose 部署，端口 19530

# 3. 部署 MinerU 服务（监听 127.0.0.1:8765，接口 /api/v4）

# 4. 配置环境变量
cp .env.example .env   # 按注释填写各服务地址与 API Key

# 5. 启动两个服务
python web/api/import_service.py   # 导入服务，端口 8000
python web/api/query_service.py    # 查询服务，端口 8001
```

### 访问

- 文件导入：`http://127.0.0.1:8000/import.html`
- 知识问答：`http://127.0.0.1:8001/chat.html`

先上传文档并等待处理完成，再进入问答页提问。

## 目录结构

```
web/api/                      # FastAPI 服务（导入 8000 / 查询 8001）
web/page/                     # 前端页面（import.html / chat.html）
processor/import_processor/   # 导入流水线（解析→分块→向量化→入库）
processor/query_processor/    # 查询流水线（改写→三路检索→RRF→重排→生成）
config/                       # 各组件配置（读取 .env）
utils/                        # Milvus / MinIO / Mongo / SSE 等工具封装
```

## 常见问题

- **上传报 500（`DATA_BASED_ROOT_DIR ... NoneType`）**：`.env` 未配置 `DATA_BASED_ROOT_DIR` 或未重启服务。
- **重新上传同名文件报 `WinError 183`**：使用已修复版本（重命名采用覆盖式 `replace`）。
- **大文档解析慢**：图片逐张调用多模态 LLM 描述，受 API 限速影响属正常现象。
- **中间件连接失败**：核对 `.env` 中 Milvus / Mongo / MinIO 的 IP 与端口。

## 安全提醒

- `.env` 含密钥，已被 `.gitignore` 排除，严禁提交仓库，仅提交 `.env.example` 模板。
- 若密钥曾泄露，请立即到对应平台重置。
