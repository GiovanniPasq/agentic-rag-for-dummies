import os
import uuid
import config
from db.vector_db_manager import VectorDbManager
from db.parent_store_manager import ParentStoreManager
from document_chunker import DocumentChuncker
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph


def _create_llm():
    """Create an LLM instance based on the active provider configuration."""
    active_config = config.LLM_CONFIGS[config.ACTIVE_LLM_CONFIG]
    model = active_config["model"]
    temperature = active_config["temperature"]

    if config.ACTIVE_LLM_CONFIG == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature, base_url=active_config.get("url"))

    elif config.ACTIVE_LLM_CONFIG == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)

    elif config.ACTIVE_LLM_CONFIG == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature)

    elif config.ACTIVE_LLM_CONFIG == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    elif config.ACTIVE_LLM_CONFIG == "minimax":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=active_config.get("base_url", "https://api.minimax.io/v1"),
            api_key=os.environ.get("MINIMAX_API_KEY"),
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {config.ACTIVE_LLM_CONFIG}")


class RAGSystem:

    def __init__(self, collection_name=config.CHILD_COLLECTION):
        self.collection_name = collection_name
        self.vector_db = VectorDbManager()
        self.parent_store = ParentStoreManager()
        self.chunker = DocumentChuncker()
        self.agent_graph = None
        self.thread_id = str(uuid.uuid4())
        self.recursion_limit = 50

    def initialize(self):
        self.vector_db.create_collection(self.collection_name)
        collection = self.vector_db.get_collection(self.collection_name)

        llm = _create_llm()
        tools = ToolFactory(collection).create_tools()
        self.agent_graph = create_agent_graph(llm, tools)
        
    def get_config(self):
        return {"configurable": {"thread_id": self.thread_id}, "recursion_limit": self.recursion_limit}
    
    def reset_thread(self):
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"Warning: Could not delete thread {self.thread_id}: {e}")
        self.thread_id = str(uuid.uuid4())