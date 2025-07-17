# PortfolioGenAI
A project to practice GenAI, including: LangChain, LlamaIndex, HuggingFace Transformers, RAG, LLM. 

## ✅ Project Structure
PortfolioGenAI/
├── src/
│   └── pgai/
│       ├── __init__.py
│       ├── langchain_helper.py
│       └── vector_db_clients/
│           ├── __init__.py
│           ├── base.py                # BaseVectorDBClient (abstract)
│           ├── faiss_client.py        # FAISSClient
│           ├── chroma_client.py       # ChromaClient
│           ├── weaviate_client.py     # WeaviateClient (future)
│           ├── pinecone_client.py     # PineconeClient (future)
│           └── postgres_client.py     # PostgresPgvectorClient (future)
├── notebooks/
│   └── example_notebook.ipynb      # optional: dev / demo space
├── data/
│   └── <your .txt reports go here>
├── pyproject.toml
├── README.md
└── .gitignore


## When developing, need to rerun `pip install -e .` or `uv pip install -e .` if:
|Action                             |Need to rerun?|
|-----------------------------------|--------------|
|Edited .py files only (code logic) |❌ No        |
|Renamed/moved modules or folders   |✅ Yes       |
|Updated pyproject.toml (e.g., src/)|✅ Yes       |	
|Added/removed dependencies         |✅ Yes       |

