import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

class VectorDbManager:
    __client: QdrantClient
    __dense_embeddings: HuggingFaceEmbeddings
    __sparse_embeddings: FastEmbedSparse

    def __init__(self):
        client_kwargs = {"prefer_grpc": config.QDRANT_PREFER_GRPC}
        if config.QDRANT_URL:
            client_kwargs.update(
                {
                    "url": config.QDRANT_URL,
                    "api_key": config.QDRANT_API_KEY or None,
                }
            )
        else:
            client_kwargs["path"] = config.QDRANT_DB_PATH

        self.__client = QdrantClient(**client_kwargs)
        self.__dense_embeddings = HuggingFaceEmbeddings(model_name=config.DENSE_MODEL)
        self.__sparse_embeddings = FastEmbedSparse(model_name=config.SPARSE_MODEL)
        self.__embedding_size = len(self.__dense_embeddings.embed_query("test"))

    def create_collection(self, collection_name):
        if not self.__client.collection_exists(collection_name):
            print(f"Creating collection: {collection_name}...")
            self.__client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.__embedding_size,
                    distance=qmodels.Distance.COSINE,
                ),
                sparse_vectors_config={config.SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()},
            )
            print(f"✓ Collection created: {collection_name}")
        else:
            print(f"✓ Collection already exists: {collection_name}")

    def delete_collection(self, collection_name):
        try:
            if self.__client.collection_exists(collection_name):
                print(f"Removing existing Qdrant collection: {collection_name}")
                self.__client.delete_collection(collection_name)
        except Exception as e:
            print(f"Warning: could not delete collection {collection_name}: {e}")

    def get_collection(self, collection_name) -> QdrantVectorStore:
        try:
            return QdrantVectorStore(
                    client=self.__client,
                    collection_name=collection_name,
                    embedding=self.__dense_embeddings,
                    sparse_embedding=self.__sparse_embeddings,
                    retrieval_mode=RetrievalMode.HYBRID,
                    sparse_vector_name=config.SPARSE_VECTOR_NAME
                )
        except Exception as e:
            raise RuntimeError(f"Unable to get collection {collection_name}: {e}") from e
