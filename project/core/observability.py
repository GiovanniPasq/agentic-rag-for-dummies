import logging
import os

import config

logger = logging.getLogger(__name__)


class Observability:

    def __init__(self):
        self.__handler = None
        self.__enabled = config.LANGFUSE_ENABLED

        if not self.__enabled:
            return

        if not config.LANGFUSE_PUBLIC_KEY or not config.LANGFUSE_SECRET_KEY:
            logger.warning("Langfuse enabled but API keys are missing — skipping")
            self.__enabled = False
            return

        try:
            from langfuse.langchain import CallbackHandler

            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", config.LANGFUSE_PUBLIC_KEY)
            os.environ.setdefault("LANGFUSE_SECRET_KEY", config.LANGFUSE_SECRET_KEY)
            os.environ.setdefault("LANGFUSE_HOST", config.LANGFUSE_BASE_URL)

            self.__handler = CallbackHandler()
            logger.info("Langfuse tracing active — sending to %s", config.LANGFUSE_BASE_URL)

        except ImportError:
            logger.warning("langfuse package not installed — tracing disabled")
            self.__enabled = False
        except Exception as exc:
            logger.warning("Could not initialize Langfuse: %s", exc)
            self.__enabled = False

    def get_handler(self):
        return self.__handler

    def flush(self):
        if self.__handler is not None:
            try:
                self.__handler.flush()
            except Exception:
                pass
