from pgai.vector_db_clients.base import BaseVectorDBClient
from langchain_huggingface import HuggingFaceEmbeddings

# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
# import chromadb
# from chromadb.config import Settings


class ChromaClient(BaseVectorDBClient):
    """easy to run locally, good for quick testing"""

    def __init__(
        self,
        collection_name="default",
        emb_model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,  # 1536,
        persist_directory="db",
    ):
        # self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        # self.collection = self.client.get_or_create_collection(name=collection_name)
        self.db = None
        self.embedding_model = HuggingFaceEmbeddings(model_name=emb_model_name)
        self.embedding_dim = embedding_dim
        self.persist_directory = persist_directory

    def from_docs(self, docs):
        # Embed & create FAISS index
        self.db = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory,
        )
        # TODO: print(f"FAISS index has {self.db.index.ntotal} vectors")
        return self

    def query(self, query_text: str, k: int = 3):
        if not self.db:
            raise ValueError("ChromaDB not initialized")
        retriever = self.db.as_retriever(search_kwargs={"k": k})
        print(retriever.search_type)
        print(retriever.search_kwargs)

        docs = retriever.get_relevant_documents(query_text)
        print(f"len(docs): {len(docs)}")
        return docs

    def save(self, path: str):
        # FIXME:path isn't used bc def persist_directory in __init__
        if not self.db:
            raise ValueError("FAISS database not initialized")
        self.db.persist()

    def load(self, path: str):
        self.db = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )

    def add(self, ids, texts, embeddings):
        assert all(len(e) == self.embedding_dim for e in embeddings), (
            "Embedding dim mismatch"
        )
        self.collection.add(documents=texts, embeddings=embeddings, ids=ids)

    def delete(self, ids):
        self.collection.delete(ids=ids)

    def __del__(self):
        self.db.delete_collection()
        self.db.persist()  # update the db on the disk
