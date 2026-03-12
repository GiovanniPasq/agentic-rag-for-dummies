"""Unit tests for multi-provider LLM configuration."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add project directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

# Mock ALL heavy dependencies before importing anything from the project.
# This avoids the need to install the full dependency tree for unit testing.
_MOCK_MODULES = [
    "langchain_huggingface", "langchain_qdrant", "langchain_qdrant.fastembed_sparse",
    "langchain_qdrant.qdrant", "qdrant_client", "qdrant_client.http",
    "qdrant_client.http.models", "langchain_ollama", "langchain_openai",
    "langchain_text_splitters", "langchain.text_splitter", "gradio",
    "pymupdf", "pymupdf.layout", "pymupdf4llm",
    "langgraph", "langgraph.graph", "langgraph.checkpoint",
    "langgraph.checkpoint.memory", "langgraph.prebuilt", "langgraph.types",
    "langchain_core", "langchain_core.tools", "langchain_core.prompts",
    "langchain_core.output_parsers", "langchain_core.messages",
    "langchain_core.runnables", "langchain_core.pydantic_v1",
    "langchain", "langchain.text_splitter",
    "pydantic", "tiktoken", "loguru",
]

for mod_name in _MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()

# Now safe to import project modules
import config
from core.rag_system import _create_llm


class TestLLMConfigs:
    """Test LLM configuration structure."""

    def test_all_providers_have_model(self):
        for provider, cfg in config.LLM_CONFIGS.items():
            assert "model" in cfg, f"Provider '{provider}' missing 'model' key"

    def test_all_providers_have_temperature(self):
        for provider, cfg in config.LLM_CONFIGS.items():
            assert "temperature" in cfg, f"Provider '{provider}' missing 'temperature' key"

    def test_minimax_config_defaults(self):
        minimax = config.LLM_CONFIGS["minimax"]
        assert minimax["model"] == "MiniMax-M2.5"
        assert minimax["base_url"] == "https://api.minimax.io/v1"
        assert minimax["temperature"] == 1.0

    def test_ollama_is_default(self):
        assert config.ACTIVE_LLM_CONFIG == "ollama"

    def test_legacy_config_matches_active(self):
        active = config.LLM_CONFIGS[config.ACTIVE_LLM_CONFIG]
        assert config.LLM_MODEL == active["model"]
        assert config.LLM_TEMPERATURE == active["temperature"]

    def test_minimax_models_available(self):
        minimax = config.LLM_CONFIGS["minimax"]
        assert minimax["model"] in ("MiniMax-M2.5", "MiniMax-M2.5-highspeed")

    def test_five_providers_configured(self):
        expected = {"ollama", "openai", "anthropic", "google", "minimax"}
        assert set(config.LLM_CONFIGS.keys()) == expected


class TestCreateLLM:
    """Test _create_llm factory function."""

    @patch("core.rag_system.config")
    def test_ollama_provider(self, mock_config):
        mock_config.ACTIVE_LLM_CONFIG = "ollama"
        mock_config.LLM_CONFIGS = {
            "ollama": {"model": "test-model", "temperature": 0, "url": "http://localhost:11434"}
        }
        mock_chat = MagicMock()
        with patch.dict(sys.modules, {"langchain_ollama": MagicMock(ChatOllama=mock_chat)}):
            _create_llm()
            mock_chat.assert_called_once_with(
                model="test-model", temperature=0, base_url="http://localhost:11434"
            )

    @patch("core.rag_system.config")
    def test_minimax_provider(self, mock_config):
        mock_config.ACTIVE_LLM_CONFIG = "minimax"
        mock_config.LLM_CONFIGS = {
            "minimax": {
                "model": "MiniMax-M2.5",
                "base_url": "https://api.minimax.io/v1",
                "temperature": 1.0,
            }
        }
        mock_chat = MagicMock()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"langchain_openai": MagicMock(ChatOpenAI=mock_chat)}):
                _create_llm()
                mock_chat.assert_called_once_with(
                    model="MiniMax-M2.5",
                    temperature=1.0,
                    base_url="https://api.minimax.io/v1",
                    api_key="test-key",
                )

    @patch("core.rag_system.config")
    def test_openai_provider(self, mock_config):
        mock_config.ACTIVE_LLM_CONFIG = "openai"
        mock_config.LLM_CONFIGS = {
            "openai": {"model": "gpt-4o-mini", "temperature": 0}
        }
        mock_chat = MagicMock()
        with patch.dict(sys.modules, {"langchain_openai": MagicMock(ChatOpenAI=mock_chat)}):
            _create_llm()
            mock_chat.assert_called_once_with(model="gpt-4o-mini", temperature=0)

    @patch("core.rag_system.config")
    def test_unsupported_provider_raises(self, mock_config):
        mock_config.ACTIVE_LLM_CONFIG = "unsupported"
        mock_config.LLM_CONFIGS = {
            "unsupported": {"model": "test", "temperature": 0}
        }
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            _create_llm()

    @patch("core.rag_system.config")
    def test_minimax_base_url_default(self, mock_config):
        mock_config.ACTIVE_LLM_CONFIG = "minimax"
        mock_config.LLM_CONFIGS = {
            "minimax": {
                "model": "MiniMax-M2.5",
                "temperature": 1.0,
            }
        }
        mock_chat = MagicMock()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"langchain_openai": MagicMock(ChatOpenAI=mock_chat)}):
                _create_llm()
                call_kwargs = mock_chat.call_args[1]
                assert call_kwargs["base_url"] == "https://api.minimax.io/v1"
