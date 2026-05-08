# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SuperBizAgent is an enterprise intelligent dialogue and AIOps assistant built with FastAPI + LangChain + LangGraph. It provides RAG (Retrieval-Augmented Generation) knowledge base Q&A and AIOps intelligent diagnosis capabilities.

**Tech Stack**: FastAPI, LangChain, LangGraph, DashScope (Qwen LLM), Milvus vector database, MCP (Model Context Protocol).

**Language**: Python 3.11+ (configured in pyproject.toml).

## Development Commands

### Environment Setup

```bash
# Install dependencies (production)
pip install -e .

# Install dependencies (development)
pip install -e ".[dev]"

# Or using uv (recommended)
pip install uv
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
uv pip install -e .
```

### Running the Application

```bash
# One-shot initialization (Docker Milvus + services + document upload)
make init

# Start all services (MCP servers + FastAPI) — background
make start

# Stop all services
make stop

# Restart all services
make restart

# Development mode with hot reload (foreground)
make dev
# Equivalent to: python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 9900

# Production mode (foreground)
make run
```

**Windows users**: `make` is not available. Use `start-windows.bat` and `stop-windows.bat` instead, or execute the PowerShell steps documented in README.md.

### Code Quality

```bash
# Format code
make format
# Runs: ruff check --select I --fix app/ && ruff format app/

# Lint code
make lint
# Runs: ruff check app/

# Auto-fix issues
make fix

# Type check
make type-check
# Runs: mypy app/ --ignore-missing-imports

# Run all checks
make check-all
# Runs: format + lint + test
```

### Testing

```bash
# Run all tests with coverage
make test
# Runs: python -m pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

# Quick test run (no coverage)
make test-quick
# Runs: python -m pytest tests/ -v

# View coverage report
make coverage
```

There is no dedicated `tests/` directory yet. Tests should be added following the pytest configuration in pyproject.toml (asyncio auto mode, coverage for `app/`).

### Docker / Infrastructure

```bash
# Start Milvus (and Attu, MinIO) via Docker Compose
make up

# Stop Milvus containers
make down

# Check container status
make status
```

### MCP Service Management

```bash
# Start individual MCP servers
make start-cls      # CLS log query MCP server (port 8003)
make start-monitor  # Monitor metrics MCP server (port 8004)
make start-api      # FastAPI main service (port 9900)

# Stop individual services
make stop-cls
make stop-monitor
make stop-api

# Check MCP status
make status-mcp
```

### Document Management

```bash
# Upload all Markdown documents in aiops-docs/ to Milvus
make upload

# List available documents
make list-docs

# Test upload a single file
make test-upload
```

## High-Level Architecture

### Application Entry

`app/main.py` bootstraps the FastAPI application. It registers routers under `/api`, mounts static files at `/static`, and manages the application lifespan (connecting to Milvus on startup, closing on shutdown).

### Layered Structure

1. **API Layer** (`app/api/`): FastAPI routers handling HTTP requests.
   - `chat.py`: Chat endpoints (regular and streaming SSE).
   - `aiops.py`: AIOps diagnosis endpoint.
   - `file.py`: Document upload and indexing.
   - `health.py`: Health check.

2. **Service Layer** (`app/services/`): Core business logic.
   - `rag_agent_service.py`: RAG Agent built with LangGraph + `ChatQwen`. Supports streaming and non-streaming query modes. Uses `MemorySaver` for session persistence and implements message trimming middleware to fit context windows.
   - `aiops_service.py`: Plan-Execute-Replan workflow orchestrator using LangGraph.
   - `vector_store_manager.py`, `vector_embedding_service.py`, `vector_index_service.py`, `vector_search_service.py`: Vector database operations (Milvus).
   - `document_splitter_service.py`: Document chunking for RAG.

3. **Agent Layer** (`app/agent/`):
   - `mcp_client.py`: Global singleton `MultiServerMCPClient` with a retry interceptor (exponential backoff, max 3 retries). Provides `get_mcp_client()` and `get_mcp_client_with_retry()`.
   - `aiops/`: AIOps core logic implementing the Plan-Execute-Replan pattern.
     - `planner.py`: Generates diagnosis steps.
     - `executor.py`: Executes steps by calling MCP tools.
     - `replanner.py`: Evaluates results and decides to continue, replan, or generate final report.
     - `state.py`: Shared state definitions for the LangGraph workflow.

4. **Core Layer** (`app/core/`):
   - `llm_factory.py`: LLM instantiation.
   - `milvus_client.py`: Milvus connection manager (`milvus_manager` singleton).

5. **Tools** (`app/tools/`):
   - `knowledge_tool.py`: Knowledge base retrieval.
   - `time_tool.py`: Current time utility.

### Key Architectural Patterns

**Plan-Execute-Replan (AIOps)**
The AIOps diagnosis is implemented as a LangGraph state machine:
1. Planner generates 4–6 diagnostic steps.
2. Executor runs each step, calling MCP tools (log query, monitoring metrics).
3. Replanner evaluates the outcome; if more steps are needed it loops back to Executor, otherwise it produces the final report.
The graph is compiled with `MemorySaver` for checkpointing.

**RAG Agent**
The chat service uses LangGraph’s `create_agent` with `ChatQwen` (from `langchain-qwq`) as the model. It loads local tools (`retrieve_knowledge`, `get_current_time`) plus dynamically loaded MCP tools. Streaming is done via `agent.astream(..., stream_mode="messages")`. Sessions are identified by `thread_id` and persisted via `MemorySaver`.

**MCP Integration**
Two independent MCP servers live in `mcp_servers/`:
- `cls_server.py` (port 8003): Log query tools (search logs, analyze patterns).
- `monitor_server.py` (port 8004): Monitoring tools (CPU/memory metrics, process list, historical tickets).
Both return mock data by default; production integration requires replacing mock implementations with real APIs (e.g., Tencent Cloud CLS, Prometheus).

**Configuration**
All configuration is centralized in `app/config.py` using Pydantic `BaseSettings`, reading from `.env`.
Key required environment variables:
- `DASHSCOPE_API_KEY`: Alibaba Cloud DashScope API key.
- `DASHSCOPE_API_BASE`: Defaults to Singapore endpoint unless explicitly set to `https://dashscope.aliyuncs.com/compatible-mode/v1`.
- `MILVUS_HOST` / `MILVUS_PORT`: Milvus connection.

**Global Singletons**
- `config` in `app/config.py`
- `milvus_manager` in `app/core/milvus_client.py`
- `rag_agent_service` in `app/services/rag_agent_service.py`
- `aiops_service` in `app/services/aiops_service.py`
- `_mcp_client` in `app/agent/mcp_client.py`

## Important Notes

- **Windows Development**: Use `start-windows.bat` / `stop-windows.bat` instead of `make`. PowerShell execution policy may need to be relaxed (`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`).
- **Milvus Dependency**: The application requires a running Milvus instance (via Docker Compose from `vector-database.yml`). `make up` starts it.
- **MCP Servers**: Must be running before the main FastAPI service starts, otherwise the RAG agent will fail to load MCP tools.
- **No Existing Tests**: There is currently no `tests/` directory. When adding tests, use `pytest-asyncio` (already in dev dependencies) and set `asyncio_mode = "auto"` (already configured in pyproject.toml).
- **Linting**: The project uses `ruff` for linting and formatting, `mypy` for type checking. Line length is 100.
- **Logs**: Application logs are written to `logs/app_YYYY-MM-DD.log` via Loguru.
