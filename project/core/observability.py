import logging
import os

import config

logger = logging.getLogger(__name__)

_langfuse_handler = None


def get_langfuse_handler():
    """
    Returns a Langfuse CallbackHandler if configured, or None.
    The handler is created lazily on first call and cached for reuse.
    """
    global _langfuse_handler

    if not config.LANGFUSE_ENABLED:
        return None

    if _langfuse_handler is not None:
        return _langfuse_handler

    if not config.LANGFUSE_PUBLIC_KEY or not config.LANGFUSE_SECRET_KEY:
        logger.warning("Langfuse enabled but API keys are missing — skipping")
        return None

    try:
        from langfuse.langchain import CallbackHandler

        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", config.LANGFUSE_PUBLIC_KEY)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", config.LANGFUSE_SECRET_KEY)
        os.environ.setdefault("LANGFUSE_HOST", config.LANGFUSE_BASE_URL)

        _langfuse_handler = CallbackHandler()
        logger.info("Langfuse tracing active — sending to %s", config.LANGFUSE_BASE_URL)
        return _langfuse_handler

    except ImportError:
        logger.warning("langfuse package not installed — tracing disabled")
        return None
    except Exception as exc:
        logger.warning("Could not initialize Langfuse: %s", exc)
        return None


def flush_langfuse():
    """Flush pending traces to Langfuse. No-op if Langfuse is not active."""
    if _langfuse_handler is not None:
        try:
            _langfuse_handler.flush()
        except Exception:
            pass
