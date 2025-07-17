from pgai.vector_db_clients.base import BaseVectorDBClient

from langchain.docstore.document import Document
# from langchain_core.documents import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter  # to split the report into chunks

from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings  # Alternative:  from langchain.embeddings.openai import OpenAIEmbeddings
  # Try:          from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

from langchain.vectorstores import FAISS  # Vector Database (indexes)
# from langchain_community.vectorstores import FAISS  # Vector Database (indexes)


class FAISSClient(BaseVectorDBClient):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", chunk_size=250, chunk_overlap=50):
        ''' Embeddings: "sentence-transformers/all-MiniLM-L6-v2" model is small (~80MB), fast on CPU, good for English
                Alternative: ultra-fast memory-light (~45MB): model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
                Alternative: embeddings = OpenAIEmbeddings()
                    TEST one embed string:
                        # embed_test = embeddings.embed_query("What company is the report about?")
                        # print(f'Test: len(embeddings)={len(embed_test)}, embeddings[:5]={embed_test[:5]}')  # Should return a 384-dim vector
        '''
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.db = None

    def from_txt_file(self, txt_path: str):
        ''' Read raw text file into a string and split into chunks '''
        with open(txt_path, 'r', encoding='utf-8') as file:
            report = file.read()
        print(f"Report loaded: {len(report)} characters")
        print("First 200 characters of the report:", report[:200] + "...")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)  # initilize the text splitter
        documents = [Document(page_content=report)]  # wrap the report string in a Document
        documents = splitter.split_documents(documents)  # split report into overlapping chunks
        print(f"Split into {len(documents)} chunks")
        # print(f'Test: docs[0].page_content:\n{docs[0].page_content}')
        # for i, doc in enumerate(docs):
        #     print(f'Test: docs[{i}].page_content:\n{doc.page_content}')

        # Embed & create FAISS index
        self.db = FAISS.from_documents(documents, self.embedding_model)
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
