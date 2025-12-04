from typing import List
from langchain_core.tools import tool
from db.parent_store_manager import ParentStoreManager

class ToolFactory:
    
    def __init__(self, collection):
        self.collection = collection
        self.parent_store_manager = ParentStoreManager()
    
    def _search_child_chunks(self, query: str, k: int) -> List[dict]:
        """Search for the top K most relevant child chunks.
        
        Args:
            query: Search query string
            k: Number of results to return
        """
        try:
            results = self.collection.similarity_search(query, k=k, score_threshold=0.7)
            return [
                {
                    "content": doc.page_content,
                    "parent_id": doc.metadata.get("parent_id", ""),
                    "source": doc.metadata.get("source", "")
                }
                for doc in results
            ]
        except Exception as e:
            print(f"Error searching child chunks: {e}")
            return []
    
    def _retrieve_parent_chunks(self, parent_ids: List[str]) -> List[dict]:
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_ids: List of parent chunk IDs to retrieve
        """
        return self.parent_store_manager.load_many(parent_ids)
    
    def create_tools(self) -> List:
        """Crea e restituisce la lista di tools."""
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        
        return [search_tool, retrieve_tool]