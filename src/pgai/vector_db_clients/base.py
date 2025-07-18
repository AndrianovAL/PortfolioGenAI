from abc import ABC, abstractmethod
from typing import List


class BaseVectorDBClient(ABC):
    @abstractmethod
    def from_docs(self, docs) -> "BaseVectorDBClient":
        """Build vector DB from a text file"""
        pass

    @abstractmethod
    def query(self, query_text: str, k: int = 3) -> List[str]:
        """Search the DB and return top-k matching chunks (as strings or documents)"""
        pass

    # @abstractmethod
    # def add(self, ids: List[str], texts: List[str], embeddings: List[List[float]]):
    #     """Add precomputed embeddings (optional for FAISS/Chroma)"""
    #     pass

    # @abstractmethod
    # def delete(self, ids: List[str]):
    #     """Delete items by ID"""
    #     pass

    @abstractmethod
    def save(self, path: str):
        """Persist the DB to disk"""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load the DB from disk"""
        pass


if __name__ == "__main__":
    db_choice = "faiss"

    if db_choice == "faiss":
        client = FAISSClient()

    client.from_txt_file("data/new_report_IBM.txt")

    results = client.query("What is IBM's net zero plan?", k=2)
    for r in results:
        print(r.page_content)

    # Optionally save the DB
    client.save("data/vector_dbs/faiss_index")

    # if db_choice == "chroma":
    #     client = ChromaClient(collection_name="companies")
    #     client.add(
    #         ids=["1", "2"],
    #         texts=["Company A does solar panels", "Company B builds wind turbines"],
    #         embeddings=[[0.1]*1536, [0.2]*1536]
    #     )
    #     result = client.query([0.1]*1536, top_k=2)
    #     print(result)
    #     client.delete(["1"])

    # TODO: Benchmark Metrics, using time.perf_counter() or timeit:
    # Insertion Time	For 1k+ vectors
    # Query Latency (ms)	With 1, 10, 100 queries
