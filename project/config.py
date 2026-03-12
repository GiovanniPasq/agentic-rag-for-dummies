import os

# --- Directory Configuration ---
_BASE_DIR = os.path.dirname(__file__)

MARKDOWN_DIR = os.path.join(_BASE_DIR, "markdown_docs")
PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store")
QDRANT_DB_PATH = os.path.join(_BASE_DIR, "qdrant_db")

# --- Qdrant Configuration ---
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# --- Model Configuration ---
DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"
SPARSE_MODEL = "Qdrant/bm25"

# --- Multi-Provider LLM Configuration ---
LLM_CONFIGS = {
    "ollama": {
        "model": "qwen3:4b-instruct-2507-q4_K_M",
        "url": "http://localhost:11434",
        "temperature": 0
    },
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0
    },
    "anthropic": {
        "model": "claude-sonnet-4-5-20250929",
        "temperature": 0
    },
    "google": {
        "model": "gemini-2.5-flash",
        "temperature": 0
    },
    "minimax": {
        "model": "MiniMax-M2.5",
        "base_url": "https://api.minimax.io/v1",
        "temperature": 1.0
    }
}

# Switch providers by changing this single line
ACTIVE_LLM_CONFIG = "ollama"

# Legacy single-provider config (kept for backward compatibility)
LLM_MODEL = LLM_CONFIGS[ACTIVE_LLM_CONFIG]["model"]
LLM_TEMPERATURE = LLM_CONFIGS[ACTIVE_LLM_CONFIG]["temperature"]

# --- Agent Configuration ---
MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 10
BASE_TOKEN_THRESHOLD = 2000
TOKEN_GROWTH_FACTOR = 0.9

# --- Text Splitter Configuration ---
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 4000
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]
