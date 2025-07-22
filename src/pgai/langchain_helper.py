from dotenv import load_dotenv
import os
from getpass import getpass
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter  # to split the report into chunks
from langchain_huggingface import HuggingFaceEmbeddings  # Alternative:  from langchain.embeddings.openai import OpenAIEmbeddings
    # Try:          from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS  # Vector Database (indexes)
from langchain.prompts import HumanMessagePromptTemplate


class CompanyScore(BaseModel):
        Company: str = Field(..., description="The name of the company")
        MetricID: int = Field(..., description="The ID of the metric")
        Score: int = Field(..., description="Score must be 1, 2, or 3")  # Accepts only int values 1, 2, or 3
        Reason1: str = Field(..., description="First reason for the score")
        Reason2: str = Field(..., description="Second reason for the score")
        Reason3: str = Field(..., description="Third reason for the score")


def get_llm(AI_Provider):
    # define the LLM name based on the AI_Provider
    if AI_Provider == "GOOGLE":
        llm_model = "gemini-2.0-flash"
    elif AI_Provider == "OPENAI":
        llm_model = "gpt-4.1-nano"  # less expensive than gpt-4o-mini

    # load the API key from the environment or user input
    load_dotenv()  # Try to load local .env file (for local dev); silently skip if not found (for CI)
    os.environ[f"{AI_Provider}_API_KEY"] = os.getenv(f"{AI_Provider}_API_KEY") or getpass(f"Enter {AI_Provider} API Key: ")  # Get API key from environment or user input
    if os.getenv(f"{AI_Provider}_API_KEY") is None:
        raise ValueError(f"❌ {AI_Provider}_API_KEY not found. Make sure it's in your .env file or set as a GitHub Action secret.")
    else:
        print(f"✅ {AI_Provider}_API_KEY loaded successfully (not printing it for security).")

    # define the LLM class based on the AI_Provider
    if AI_Provider == "GOOGLE":
        from langchain_google_genai import ChatGoogleGenerativeAI as ChatLLM
    elif AI_Provider == "OPENAI":
        from langchain_openai import ChatOpenAI as ChatLLM

    # greate the LLM instance
    llm = ChatLLM(temperature=0.0, model=llm_model)  # For normal accurate responses
    # print("My LLM version is:", llm.invoke("What LLM version are you?").content)

    # structure the LLM output to the CompanyScore class
    llm_structured = llm.with_structured_output(CompanyScore)

    return llm, llm_structured

def create_vectordb_from_txt_file(txt_filename: str, chunk_size: int = 250, chunk_overlap: int = 50) -> FAISS:
    with open(txt_filename, 'r', encoding='utf-8') as f:
        report = f.read()

    print(f"✅ Report loaded ({len(report)} characters).")
    print(f"Sample:\n{report[:200]}...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents([Document(page_content=report)])

    print(f"✅ Document split into {len(docs)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    print(f"✅ Vector DB created with {db.index.ntotal} vectors.")

    return db

def split_txt_to_docs(txt_path, chunk_size=250, chunk_overlap=50):
    ''' Read raw text file into a string and split into chunks '''
    with open(txt_path, 'r', encoding='utf-8') as file:
        report = file.read()
    print(f"Report loaded: {len(report)} characters")
    print("First 200 characters of the report:", report[:200] + "...")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # initilize the text splitter
    documents = [Document(page_content=report)]  # wrap the report string in a Document
    documents = splitter.split_documents(documents)  # split report into overlapping chunks
    print(f"Split into {len(documents)} chunks")
    # print(f'Test: docs[0].page_content:\n{docs[0].page_content}')
    # for i, doc in enumerate(docs):
    #     print(f'Test: docs[{i}].page_content:\n{doc.page_content}')

    return documents

# def create_vectordb_from_txt_file(txt_filename: str) -> FAISS:
#     chunk_size = 250  # FIXME: calibrate
#     chunk_overlap = 50  # FIXME: calibrate

#     # Read the IBM report file into a string
#     with open('data/'+txt_filename, 'r', encoding='utf-8') as file:
#         report = file.read()
#     print(f"Report loaded successfully! It's length: {len(report)} characters")
#     print("First 200 characters of the report:", report[:200] + "...")

#     # split the report into overlapping chunks
#     documents = [Document(page_content=report)]  # wrap the report string in a Document
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # initilize the text splitter
#     docs = text_splitter.split_documents(documents)  # split report into overlapping chunks
#     print(f'The number of chunks is {len(docs)}')
#     # # test
#     # print(f'Test: docs[0].page_content:\n{docs[0].page_content}')
#     # for i, doc in enumerate(docs):
#     #     print(f'Test: docs[{i}].page_content:\n{doc.page_content}')    return db # TODO: Implement this function

#     # define how to map a string (can be a sentence or a paragraph) to a vector
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # This model is small (~80MB), fast on CPU, good for English # Alternative: ultra-fast memory-light (~45MB): model_name="sentence-transformers/paraphrase-MiniLM-L3-v2" # Alternative: embeddings = OpenAIEmbeddings()
#     # # Test: one embed string
#     # embed_test = embeddings.embed_query("What company is the report about?")
#     # print(f'Test: len(embeddings)={len(embed_test)}, embeddings[:5]={embed_test[:5]}')  # Should return a 384-dim vector

#     # create a DB of vector embeddings from the docs
#     db = FAISS.from_documents(docs, embeddings)
#     print(f'The number of vectors in the DB is {db.index.ntotal}')

#     return db


def prep_similarity_search_query(metrics, metric_id: int) -> str:
    metric_row = metrics.loc[metrics.MetricID == metric_id].iloc[0]
    prompt = HumanMessagePromptTemplate.from_template(
        """Metric ID {metric_id}:\n{metric_description}""",
        input_variables=["metric_id", "metric_description"]
    )
    return prompt.format(metric_id=metric_id, metric_description=metric_row["MetricDescription"]).content



def similarity_search(query: str, db, top_k: int = 4) -> str:
    docs = db.similarity_search(query, k=top_k)
    return " ".join(doc.page_content for doc in docs)


def prep_prompt_inputs(new_company, metrics, metric_id, train_examples, report_summary) -> dict:
    metric_row = metrics.loc[metrics.MetricID == metric_id].iloc[0]
    examples = train_examples.loc[train_examples.MetricID == metric_id].reset_index(drop=True)

    if len(examples) < 3:
        print(f"⚠️ Only {len(examples)} examples found for Metric ID {metric_id}. Padding with N/A.")
        # Pad with empty rows if less than 3 examples
        for _ in range(3 - len(examples)):
            examples = pd.concat([examples, pd.Series({
                'Company': "N/A", 'Score': "N/A",
                'Reason1': "N/A", 'Reason2': "N/A", 'Reason3': "N/A"
            }).to_frame().T], ignore_index=True)

    prompt_inputs = {
        "new_company": new_company,
        "metric_id": metric_id,
        "metric_description": metric_row["MetricDescription"],
        "new_company_report_chunks_summary": report_summary
    }

    for idx, row in examples.head(3).iterrows():
        idx1 = idx + 1
        prompt_inputs.update({
            f"Company_{idx1}": row.Company,
            f"Score_{idx1}": row.Score,
            f"Reason1_{idx1}": row.Reason1,
            f"Reason2_{idx1}": row.Reason2,
            f"Reason3_{idx1}": row.Reason3
        })

    return prompt_inputs


if __name__ == "__main__":
    AI_Provider = "GOOGLE"
    # LLM_Provider = "OPENAI"

    llm, llm_structured = get_llm(AI_Provider)
    print("My LLM version is:", llm.invoke("What LLM version are you?").content)
