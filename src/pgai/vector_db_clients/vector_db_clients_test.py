from pgai.vector_db_clients.chroma_client import ChromaClient
from pgai.vector_db_clients.faiss_client import FAISSClient
from pgai import langchain_helper


if __name__ == "__main__":
    db_choice = "chroma"

    if db_choice == "faiss":
        client = FAISSClient()
    elif db_choice == "chroma":
        client = ChromaClient()

    docs = langchain_helper.split_txt_to_docs(txt_path="data/new_report_IBM.txt")
    client.from_docs(docs)

    print(f"\n=== TEST: {db_choice} ===")
    results = client.query("What is IBM's net zero plan?", k=2)
    print("For query 'What is IBM's net zero plan?' received the following results:")
    for num, r in enumerate(results):
        print(f"{num}: {r.page_content}")

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
