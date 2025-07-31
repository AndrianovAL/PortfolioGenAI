from pgai.vector_db_clients.base import BaseVectorDBClient

# from langchain.docstore.document import Document
# from langchain_core.documents import Document

# from langchain.text_splitter import RecursiveCharacterTextSplitter  # split to chunks

from langchain_huggingface import HuggingFaceEmbeddings
# Depricated: from langchain.embeddings import HuggingFaceEmbeddings
# Alternative: from langchain.embeddings import OpenAIEmbeddings # from langchain.embeddings.openai import OpenAIEmbeddings
# Try:         from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

# from langchain.vectorstores import FAISS  # Vector Database (indexes)
from langchain_community.vectorstores import FAISS  # Vector Database (indexes)


class FAISSClient(BaseVectorDBClient):
    def __init__(self, emb_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Embeddings: "sentence-transformers/all-MiniLM-L6-v2" model is small (~80MB), fast on CPU, good for English
        Alternative: ultra-fast memory-light (~45MB): model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
        Alternative: embeddings = OpenAIEmbeddings()
        TEST one embed string:
            # embed_test = embeddings.embed_query("What company is the report about?")
            # print(f'Test: len(embeddings)={len(embed_test)}, embeddings[:5]={embed_test[:5]}')  # Should return a 384-dim vector
        """
        self.embedding_model = HuggingFaceEmbeddings(model_name=emb_model_name)
        self.db = None

    def from_docs(self, docs):
        # Embed & create FAISS index
        self.db = FAISS.from_documents(docs, self.embedding_model)
        print(f"FAISS index has {self.db.index.ntotal} vectors")

        return self

    def query(self, query_text: str, k: int = 3):
        if not self.db:
            raise ValueError("FAISS database not initialized")
        return self.db.similarity_search(query_text, k=k)

    def save(self, path: str):
        if not self.db:
            raise ValueError("FAISS database not initialized")
        self.db.save_local(path)

    def load(self, path: str):
        self.db = FAISS.load_local(path, self.embedding_model)
