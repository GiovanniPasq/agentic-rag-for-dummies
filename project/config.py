# config.py
"""
Central configuration file for all constants and settings.
"""

import os

# --- Directory Configuration ---
DOCS_DIR = "docs"
PARENT_STORE_PATH = "parent_store"
QDRANT_DB_PATH = "qdrant_db"

# --- Qdrant Configuration ---
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"  # Name to use for sparse vectors in Qdrant

# --- Model Configuration ---
# Dense Embeddings (for semantic understanding)
DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"
# Sparse Embeddings (for keyword matching)
SPARSE_MODEL = "Qdrant/bm25"
# LLM (Example with Ollama)
LLM_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
LLM_TEMPERATURE = 0.1
# For Google Gemini (uncomment if needed)
# os.environ["GOOGLE_API_KEY"] = "your-api-key-here"
# GEMINI_MODEL = "gemini-2.0-flash-exp"

# --- Text Splitter Configuration ---
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]