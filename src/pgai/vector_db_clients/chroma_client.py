from pgai.vector_db_clients.base import BaseVectorDBClient
import chromadb
from chromadb.config import Settings


class ChromaClient(BaseVectorDBClient):
    ''' easy to run locally, good for quick testing '''
    def __init__(self, collection_name="default", embedding_dim=1536):
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_dim = embedding_dim

    def add(self, ids, texts, embeddings):
        assert all(len(e) == self.embedding_dim for e in embeddings), "Embedding dim mismatch"
        self.collection.add(documents=texts, embeddings=embeddings, ids=ids)

    def query(self, query_embedding, top_k=3):
        assert len(query_embedding) == self.embedding_dim, "Query embedding dim mismatch"
        return self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

    def delete(self, ids):
        self.collection.delete(ids=ids)
    pass