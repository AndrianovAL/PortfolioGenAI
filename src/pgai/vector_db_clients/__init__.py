# Leave empty or use to simplify imports like from pgai.vector_db_clients import FAISSClient

# You can add other vector DB clients here as you implement them
from .faiss_client import FAISSClient # allows "from pgai.vector_db_clients import FAISSClient" instead of "pgai.vector_db_clients.faiss_client import FAISSClient"
from .chroma_client import ChromaClient  # if implemented later

__all__ = ["FAISSClient", "ChromaClient"]  # allows from pgai.vector_db_clients import *
