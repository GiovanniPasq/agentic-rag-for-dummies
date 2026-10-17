#!/usr/bin/env bash
set -e

# Navigate to project root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "=========================================="
echo "  🚀 Starting Agentic RAG System"
echo "=========================================="

# 1. Check/Prepare environment file
if [ ! -f "project/.env" ] && [ -f "project/.env.example" ]; then
    echo "📄 Creating project/.env from project/.env.example..."
    cp project/.env.example project/.env
fi

# 2. Informative check for Ollama
if command -v curl >/dev/null 2>&1; then
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "⚠️  Note: Ollama does not seem to be running on localhost:11434."
        echo "   If using local models (default), remember to run 'ollama serve'."
    fi
fi

# 3. Launch application (prefer uv if available)
if command -v uv >/dev/null 2>&1; then
    echo "⚡ Using uv for fast environment management..."
    uv run project/app.py
else
    echo "🐍 uv not detected, falling back to standard Python venv..."
    PYTHON_CMD=""
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    else
        echo "❌ Error: Python 3.11+ is required but neither python3 nor python was found."
        exit 1
    fi

    if [ ! -d ".venv" ]; then
        echo "🔨 Creating virtual environment in .venv..."
        $PYTHON_CMD -m venv .venv
    fi

    # Activate virtual environment
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    elif [ -f ".venv/Scripts/activate" ]; then
        source .venv/Scripts/activate
    fi

    echo "📦 Checking and installing dependencies..."
    pip install -q -r requirements.txt

    echo "🚀 Starting app..."
    python project/app.py
fi
