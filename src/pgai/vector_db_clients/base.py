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
