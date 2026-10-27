# KeeperSage Enterprise Document System

An enterprise-grade RAG knowledge base: upload PDF/MD documents, and the system automatically parses, chunks, embeds, and indexes them — then answers questions through multi-source retrieval on a web UI, with streaming responses and inline image references.

> **End-to-end, independently built**: document parsing → hybrid retrieval → multi-route fusion → reranking → streaming generation, covering the full engineering pipeline of a RAG application from data ingestion to answer delivery.

---

## Highlights

- **Complete RAG pipelines**: Both ingestion (parsing / chunking / embedding / indexing) and querying (rewriting / retrieval / fusion / reranking / generation) are orchestrated as **LangGraph** state machines with node-level streaming and observability.
- **Hybrid retrieval**: BGE-M3 produces both dense (semantic) and sparse (learned lexical) vectors, fused via weighted ranking — combining semantic generalization with precise keyword matching.
- **Three parallel retrieval routes**: vanilla vector search / **HyDE** hypothetical-document retrieval / web search via the **MCP** protocol (Alibaba Cloud Bailian), improving recall through source diversity.
- **Two-stage ranking**: Reciprocal Rank Fusion (RRF) merges multi-route results by rank, followed by Cross-Encoder reranking for precision.
- **Multimodal document parsing**: MinerU converts PDFs to Markdown; images are captioned by a multimodal LLM and embedded alongside chunks, with original images surfaced in answers.
- **Streaming Q&A**: Server-Sent Events push node-level progress and incremental answers in real time; the frontend renders answers with referenced images.

## Architecture

```
Ingestion pipeline (LangGraph)
PDF/MD → MinerU parsing → multimodal image captioning → chunking
      → embedding (dense + sparse) → Milvus indexing
      → MinIO object storage → MongoDB records

Query pipeline (LangGraph)
user query → rewriting + entity-name extraction & confirmation
          → ┌ vanilla hybrid vector search (dense + sparse)
            ├ HyDE hypothetical-document retrieval
            └ web search via MCP
          → RRF fusion → Cross-Encoder reranking
          → LLM generation (SSE streaming) → answer + images
```

## Features

- Batch upload of PDF / MD files, stored by `date / task-ID`, with real-time progress and node-level logs
- Parsed artifacts (Markdown, images) automatically structured; images stored in object storage and linked back
- Session-based multi-turn Q&A with history persisted in MongoDB
- Answers carry referenced images automatically
- Dual-page web UI (Import / Chat) with one-click navigation

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph, FastAPI, BackgroundTasks |
| Parsing | MinerU (local API), multimodal LLM captioning |
| Embedding | BGE-M3 (dense + sparse), Milvus hybrid search |
| Ranking | RRF, Cross-Encoder reranking (DashScope) |
| Generation | OpenAI-compatible LLM, SSE streaming |
| Web search | MCP protocol (Streamable HTTP) via Bailian |
| Storage | Milvus / MongoDB / MinIO |
| Frontend | Vanilla HTML/JS (no framework, zero build step) |

## Quick Start

### Requirements

- Python 3.11
- NVIDIA GPU (8GB+ VRAM recommended for BGE-M3; CPU works but slower)
- Docker: Milvus, MongoDB, MinIO
- MinerU local parsing service (or adapt to a cloud parsing API)

### Steps

```bash
# 1. Clone and install dependencies
git clone <repo-url> && cd KeeperSage
pip install -r requirements.txt

# Note: the mcp package must be 1.x (pip install "mcp<2");
# 2.x is protocol-incompatible with the Bailian MCP service.

# 2. Start middleware (Docker)
docker run -d -p 27017:27017 --name mongo mongo:6
docker run -d -p 9000:9000 -p 9001:9001 --name minio minio/minio server /data --console-address ":9001"
# Deploy Milvus standalone via its official docker-compose (port 19530)

# 3. Deploy the MinerU service (listens on 127.0.0.1:8765, API path /api/v4)

# 4. Configure environment variables
cp .env.example .env   # fill in service addresses and API keys

# 5. Start both services
python web/api/import_service.py   # ingestion service, port 8000
python web/api/query_service.py    # query service, port 8001
```

### Access

- Import page: `http://127.0.0.1:8000/import.html`
- Chat page: `http://127.0.0.1:8001/chat.html`

Upload documents and wait for processing to finish before asking questions.

## Project Layout

```
web/api/                      # FastAPI services (import 8000 / query 8001)
web/page/                     # Frontend pages (import.html / chat.html)
processor/import_processor/   # Ingestion pipeline (parse → chunk → embed → index)
processor/query_processor/    # Query pipeline (rewrite → 3-route retrieval → RRF → rerank → generate)
config/                       # Component configs (reads .env)
utils/                        # Milvus / MinIO / Mongo / SSE helpers
```

## Troubleshooting

- **500 on upload (`DATA_BASED_ROOT_DIR ... NoneType`)**: `DATA_BASED_ROOT_DIR` missing in `.env`, or the service was not restarted after editing.
- **`WinError 183` when re-uploading a same-named file**: use the fixed version (renaming now overwrites via `replace`).
- **Slow parsing of large documents**: images are captioned one by one via the multimodal LLM and subject to API rate limits.
- **Middleware connection failures**: verify Milvus / Mongo / MinIO IPs and ports in `.env`.

## Security Notes

- `.env` contains secrets and is excluded via `.gitignore` — never commit it; only commit the `.env.example` template.
- Rotate any key that may have been leaked immediately.
