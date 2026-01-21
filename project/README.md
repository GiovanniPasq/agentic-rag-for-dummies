# Agentic RAG System Documentation

An **Agentic Retrieval-Augmented Generation (RAG)** system built with **LangGraph**, featuring **parent–child chunking**, **hybrid dense + sparse retrieval**, and **multi-LLM provider support**.


## Table of Contents

[Quick Start](#quick-start) | [Architecture Overview](#architecture-overview) | [Project Structure](#project-structure) | [Configuration Guide](#configuration-guide) | [Common Customizations](#common-customizations) | [Advanced Topics](#advanced-topics) | [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Application

Start the Gradio interface locally:

```bash
python project/app.py
```

The application will be available at `http://localhost:7860` (default Gradio port).

### Prerequisites

- Python 3.10+
- Ollama (local) or API keys for OpenAI, Anthropic, or Google Gemini

---

## Architecture Overview

This system implements an advanced RAG pipeline with the following key features:

- **Parent-Child Chunking**: Documents are split into small child chunks (for precise retrieval) linked to larger parent chunks (for rich context)
- **Hybrid Search**: Combines dense embeddings and sparse (BM25) retrieval for optimal results
- **LangGraph Agent**: Orchestrates query rewriting, retrieval, and response generation
- **Multi-Provider Support**: Seamlessly switch between Ollama, OpenAI, Google Gemini, and Anthropic Claude
- **Vector Storage**: Uses Qdrant for efficient similarity search

### Data Flow

```
PDF → Markdown Conversion → Parent/Child Chunking → Vector Indexing → Agent Retrieval → LLM Response
```

---

## Project Structure

### Entry Point & Configuration

| File | Purpose |
|------|---------|
| `project/app.py` | Application entry point, launches Gradio UI |
| `project/config.py` | **Central configuration hub** - edit this for provider/model/chunking changes |
| `project/util.py` | PDF to Markdown conversion using `pymupdf4llm` |
| `project/document_chunker.py` | Parent/child splitting logic with cleaning and merging rules |
| `project/Dockerfile` | Dockerfile with Ollama for local deployment |

### Core System

| File | Purpose |
|------|---------|
| `project/core/rag_system.py` | System bootstrap - creates managers and compiles LangGraph agent |
| `project/core/document_manager.py` | Document ingestion pipeline (convert, chunk, index) |
| `project/core/chat_interface.py` | Thin wrapper for agent graph interaction |

### Database Layer

| File | Purpose |
|------|---------|
| `project/db/vector_db_manager.py` | Qdrant client wrapper with embedding initialization |
| `project/db/parent_store_manager.py` | File-backed storage for parent chunks |

### RAG Agent (LangGraph)

| File | Purpose |
|------|---------|
| `project/rag_agent/graph.py` | Graph builder and compilation logic |
| `project/rag_agent/graph_state.py` | Shared and per-agent graph state definitions and answer accumulation/reset logic|
| `project/rag_agent/nodes.py` | Node implementations (summarize, rewrite, agent execution, aggregate) |
| `project/rag_agent/edges.py` | Conditional edge routing logic (e.g., routing based on query clarity) |
| `project/rag_agent/tools.py` | Retrieval tools (`search_child_chunks`, `retrieve_parent_chunks`) |
| `project/rag_agent/prompts.py` | System prompts for agent behavior |
| `project/rag_agent/schemas.py` | Structured output schemas (Pydantic models) |

### User Interface

| File | Purpose |
|------|---------|
| `project/ui/css.py` | Custom CSS styling for the Gradio interface |
| `project/ui/gradio_app.py` | Gradio UI implementation with document upload and chat |

---

## Configuration Guide

All primary settings are in `project/config.py`. Key parameters:

### Directory Configuration

```python
MARKDOWN_DIR = "markdown_docs"        # Storage for converted PDF → Markdown files
PARENT_STORE_PATH = "parent_store"    # File-backed storage for parent chunks
QDRANT_DB_PATH = "qdrant_db"          # Local Qdrant vector database path
```

### Qdrant Configuration

```python
CHILD_COLLECTION = "document_child_chunks"  # Collection name for child chunks
SPARSE_VECTOR_NAME = "sparse"               # Named sparse vector field (BM25)
```

### Model Configuration

```python
# Default: single model configuration
DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"
SPARSE_MODEL = "Qdrant/bm25"
LLM_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
LLM_TEMPERATURE = 0  # 0 = deterministic, 1 = creative
```

### Text Splitter Configuration

```python
CHILD_CHUNK_SIZE = 500              # Size of chunks used for retrieval
CHILD_CHUNK_OVERLAP = 100           # Overlap between chunks (prevents context loss)
MIN_PARENT_SIZE = 2000              # Minimum parent chunk size
MAX_PARENT_SIZE = 10000             # Maximum parent chunk size

# Markdown header splitting strategy
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]
```

---

## Common Customizations

### 1. Switching LLM Provider (Single Provider)

> **Performance Note:** LLMs with ≥14B parameters typically offer superior reasoning, context comprehension, and response quality compared to smaller models. This applies to both proprietary and open-source models, as long as they **support native tool/function calling,** which is required for agentic RAG workflows.

If you want to permanently switch from one provider to another (e.g., Ollama → Google Gemini), follow this steps:

**Step 1:** Install the provider's SDK

```bash
pip install langchain-google-genai
```

**Step 2:** Set environment variable

```bash
export GOOGLE_API_KEY="your-google-key"
```

**Step 3:** Update `project/config.py`

```python
LLM_MODEL = "gemini-2.5-pro"
LLM_TEMPERATURE = 0
```

**Step 4:** Modify `project/core/rag_system.py`

Replace:

```python
llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
```

With:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
```

## 2. Multi-Provider Configuration

This approach allows you to maintain multiple provider configurations and switch between them easily.

**Step 1:** Install required SDKs

```bash
pip install langchain-openai langchain-anthropic langchain-google-genai
```

**Step 2:** Set environment variables

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

**Step 3:** Update `project/config.py` with multi-provider configuration

```python
# --- Multi-Provider LLM Configuration ---
LLM_CONFIGS = {
    "ollama": {
        "model": "llama3.2:3b",
        "url":"http://localhost:11434"
        "temperature": 0
    },
    "openai": {
        "model": "gpt-4o",
        "temperature": 0
    },
    "anthropic": {
        "model": "claude-sonnet-4-20250514",
        "temperature": 0
    },
    "google": {
        "model": "gemini-2.5-flash",
        "temperature": 0
    }
}

# Switch providers by changing this single line
ACTIVE_LLM_CONFIG = "ollama"
```

**Step 4:** Modify `project/core/rag_system.py` in the `initialize()` method

Replace the existing LLM initialization with:

```python
def initialize(self):
    self.vector_db.create_collection(self.collection_name)
    collection = self.vector_db.get_collection(self.collection_name)
    
    # Load active configuration
    active_config = config.LLM_CONFIGS[config.ACTIVE_LLM_CONFIG]
    model = active_config["model"]
    temperature = active_config["temperature"]
    
    if active_config == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, temperature=temperature, base_url=active_config["url"])
        
    elif active_config == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, temperature=temperature)
        
    elif active_config == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=model, temperature=temperature)
        
    elif active_config == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        
    else:
        raise ValueError(f"Unsupported LLM provider: {active_config}")
    
    # Continue with tool and graph initialization
    tools = ToolFactory(collection).create_tools()
    self.agent_graph = create_agent_graph(llm, tools)
```

**Switching Providers:** Simply change `ACTIVE_LLM_CONFIG` in `config.py`:

```python
ACTIVE_LLM_CONFIG = "google"  # Switch to Gemini Pro
# ACTIVE_LLM_CONFIG = "anthropic"  # Or Claude Sonnet
# ACTIVE_LLM_CONFIG = "openai"  # Or GPT-4o
```

---

**Provider Reference Table:**

| Provider | Environment Variable | Import Statement | Example Models |
|----------|---------------------|------------------|----------------|
| OpenAI | `OPENAI_API_KEY` | `from langchain_openai import ChatOpenAI` | `gpt-4o`, `gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `from langchain_anthropic import ChatAnthropic` | `claude-opus-4-20250514`, `claude-sonnet-4-20250514` |
| Google | `GOOGLE_API_KEY` | `from langchain_google_genai import ChatGoogleGenerativeAI` | `gemini-2.5-pro`, `gemini-2.5-flash` |
| Ollama | None (local) | `from langchain_ollama import ChatOllama` | `qwen3:4b-instruct-2507-q4_K_M`, `ministral-3:8b-instruct-2512-q4_K_M`, `llama3.1:8b-instruct-q6_K` |

---

### 2. Changing Embedding Models

**Why change?** Trade-offs between speed, cost, and quality.

**Step 1:** Update `project/config.py`

```python
# Example: Faster, smaller model
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Example: Higher quality, slower model
# DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Example: Gemma embeddings (Google's open model)
# DENSE_MODEL = "google/embeddinggemma-300m"

# Example: Qwen embeddings (Alibaba's multilingual model)
# DENSE_MODEL = "Qwen/Qwen3-Embedding-8B"

SPARSE_MODEL = "Qdrant/bm25"  # Usually no need to change
```

**Step 2:** Re-index your documents

⚠️ **Important:** Changing embeddings requires re-indexing all documents through the Gradio UI.

**Implementation Details** (in `project/db/vector_db_manager.py`):

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse
import config

self.__dense_embeddings = HuggingFaceEmbeddings(model_name=config.DENSE_MODEL)
self.__sparse_embeddings = FastEmbedSparse(model_name=config.SPARSE_MODEL)
```

**Popular Embedding Models:**

| Model | Context Size | Vector Dimension | Speed | Quality | Use Case |
|-------|--------------|------------------|-------|---------|----------|
| all-MiniLM-L6-v2 | 512 tokens | 384 | Fast | Good | General purpose, quick semantic similarity |
| all-mpnet-base-v2 | 512 tokens | 768 | Medium | Excellent | High-accuracy semantic search |
| bge-large-en-v1.5 | 512 tokens | 1024 | Slow | Best | Production-grade retrieval on GPU |
| google/embeddinggemma-300m | 2048 tokens | 768 (MRL: 512 / 256 / 128) | Fast | Very Good | Lightweight, efficient multilingual retrieval |
| Qwen/Qwen3-Embedding-8B | 32768 tokens | 4096 (configurable 32–4096) | Medium | Excellent / SOTA | Large-scale multilingual embeddings, long-context RAG |

---

### 3. Adjusting Chunking Strategy

**Why adjust?** Balance between retrieval precision and context richness.

**Step 1:** Update chunk sizes in `project/config.py`

```python
# For short, factual queries (e.g., technical documentation)
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 60
MIN_PARENT_SIZE = 1500
MAX_PARENT_SIZE = 8000

# For narrative or contextual queries (e.g., legal documents)
# CHILD_CHUNK_SIZE = 800
# CHILD_CHUNK_OVERLAP = 100
# MIN_PARENT_SIZE = 3000
# MAX_PARENT_SIZE = 15000
```

**Step 2 (Optional):** Replace the splitter in `project/document_chunker.py`

**Default (Character-based):**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

self.__child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHILD_CHUNK_SIZE,
    chunk_overlap=config.CHILD_CHUNK_OVERLAP
)
```

**Alternative (Sentence-aware):**
```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

self.__child_splitter = SentenceTransformersTokenTextSplitter(
    chunk_size=config.CHILD_CHUNK_SIZE,
    chunk_overlap=config.CHILD_CHUNK_OVERLAP
)
```

**Step 3:** Re-run ingestion pipeline

Upload documents again through the Gradio interface to apply new chunking.

**Chunking Guidelines:**

| Document Type | Child Size | Parent Size | Reasoning |
|---------------|-----------|-------------|-----------|
| Technical Docs | 300-500 | 2000-5000 | Precise lookups, code snippets |
| Legal Contracts | 600-1000 | 5000-15000 | Context-heavy, definitions |
| Research Papers | 400-600 | 3000-8000 | Balance of precision and context |
| FAQs / Knowledge Base | 200-400 | 1500-4000 | Short, focused answers |

---

## Advanced Topics

### Customizing the RAG Agent

**Location:** `project/rag_agent/`

**Add/Remove Nodes:** Edit `graph.py` and `nodes.py`

Example: Adding a fact-checking node
```python
# In nodes.py
def fact_check_node(state):
    # Your fact-checking logic
    return {"fact_checked": True}

# In graph.py
builder.add_node("fact_check", fact_check_node)
builder.add_edge("retrieve", "fact_check")
```

**Modify Conditional Routing:** Edit `edges.py` to change graph flow logic

Example from the system - routing based on query clarity:
```python
def route_after_rewrite(state: State) -> Literal["human_input", "process_question"]:
    """Routes to human input if question unclear, otherwise processes all rewritten queries"""
    if not state.get("questionIsClear", False):
        return "human_input"
    else:
        # Fan-out: send each rewritten question to parallel processing
        return [
            Send("process_question", {"question": query, "question_index": idx, "messages": []})
            for idx, query in enumerate(state["rewrittenQuestions"])
        ]
```

This pattern allows the agent to either request clarification from the user or fan-out multiple query variations for parallel retrieval.

**Modify Prompts:** Edit `prompts.py` to change agent behavior and response style

**Add Custom Tools:** Extend `tools.py` with new retrieval strategies or external integrations

### Replacing Storage Backends

**Vector Database:**
- Default: Local Qdrant
- Alternatives: Remote Qdrant Cloud, Pinecone, Weaviate
- Edit: `project/db/vector_db_manager.py`

**Parent Store:**
- Default: JSON file
- Alternatives: PostgreSQL, MongoDB, S3
- Edit: `project/db/parent_store_manager.py`

### Extending the UI

**Location:** `project/ui/gradio_app.py`

Add runtime settings, admin panels, or analytics:
```python
with gr.Accordion("Advanced Settings", open=False):
    provider_dropdown = gr.Dropdown(
        choices=["openai", "anthropic", "google", "ollama"],
        label="LLM Provider"
    )
```

### Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t rag-system -f project/Dockerfile .

# Run container
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=your-key \
  -v $(pwd)/data:/app/data \
  rag-system
```

### Performance Optimization

**Tips:**
- Use GPU-enabled embeddings for large document sets
- Implement caching for frequently retrieved chunks
- Tune `top_k` retrieval parameters in tools.py
- Consider async processing for multi-document ingestion
- Monitor Qdrant memory usage and tune collection parameters

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Model not found" error | Incorrect model name for provider | Verify `LLM_MODEL` matches provider's API (e.g., `gpt-4o-mini` not `gpt4-mini`) |
| Low-quality retrieval results | Poor embedding model or chunk configuration | Re-index with better embeddings (e.g., all-mpnet-base-v2) or adjust chunk sizes |
| Slow response times | Large embedding model or high `top_k` value | Use smaller embedding models (e.g., all-MiniLM-L6-v2) or reduce `top_k` in retrieval tools |
| API rate limits exceeded | Too many requests to external provider | Add retry logic with exponential backoff or switch to local Ollama models |
| Out of memory errors | Large document set or embedding model | Use smaller embeddings, reduce batch size, or enable GPU acceleration |
| Empty retrieval results | Collection not indexed or wrong collection name | Verify documents are uploaded and `CHILD_COLLECTION` name matches in config |
| Import errors after provider switch | Missing SDK installation | Install required package: `pip install langchain-{provider}` |
| Inconsistent answers across runs | High temperature setting | Set `LLM_TEMPERATURE = 0` in config for deterministic responses |