# from langchain_google_genai import GoogleGenerativeAI  # LLM: Google Gemini
#     # Maybe need:   from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate  # for templates
# from langchain.chains import LLMChain  # chain llm & prompt

    # llm = GoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-2.0-flash", temperature=0.2, max_tokens=6_000)
    #     # Maybe need:   llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-2.0-flash", temperature=0.2, max_tokens=500)
    #     # Alternative:  llm = OpenAI(model_name="text-davinci-003")
 
from dotenv import load_dotenv
import os
from getpass import getpass
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter  # to split the report into chunks
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings  # Alternative:  from langchain.embeddings.openai import OpenAIEmbeddings
    # Try:          from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS  # Vector Database (indexes); alternatives: Pinecon, Weaviate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate


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


def create_vectordb_from_txt_file(txt_filename: str) -> FAISS:
    chunk_size = 250  # FIXME: calibrate
    chunk_overlap = 50  # FIXME: calibrate

    # Read the IBM report file into a string
    with open('data/'+txt_filename, 'r', encoding='utf-8') as file:
        report = file.read()
    print("Report loaded successfully!")
    print(f"Report length: {len(report)} characters")
    print("First 200 characters of the report:")
    print(report[:200] + "...")

    # split the report into overlapping chunks
    documents = [Document(page_content=report)]  # wrap the report string in a Document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # initilize the text splitter
    docs = text_splitter.split_documents(documents)  # split report into overlapping chunks
    print(f'The number of chunks is {len(docs)}')
    # # test
    # print(f'Test: docs[0].page_content:\n{docs[0].page_content}')
    # for i, doc in enumerate(docs):
    #     print(f'Test: docs[{i}].page_content:\n{doc.page_content}')    return db # TODO: Implement this function

    # define how to map a string (can be a sentence or a paragraph) to a vector
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # This model is small (~80MB), fast on CPU, good for English # Alternative: ultra-fast memory-light (~45MB): model_name="sentence-transformers/paraphrase-MiniLM-L3-v2" # Alternative: embeddings = OpenAIEmbeddings()
    # # Test: one embed string
    # embed_test = embeddings.embed_query("What company is the report about?")
    # print(f'Test: len(embeddings)={len(embed_test)}, embeddings[:5]={embed_test[:5]}')  # Should return a 384-dim vector

    # create a DB of vector embeddings from the docs
    db = FAISS.from_documents(docs, embeddings)
    print(f'The number of vectors in the DB is {db.index.ntotal}')

    return db


def prep_simularity_search_query(metrics, metric_id:int) -> str:

    user_prompt = HumanMessagePromptTemplate.from_template(
        """{metric_name} metric:\n{metric_description}""",
        input_variables=["metric_name", "metric_description"]
    )

    query = user_prompt.format(
        metric_name = metrics[metrics["MetricID"]==metric_id]["MetricName"].values[0],
        metric_description = metrics[metrics["MetricID"]==metric_id]["MetricDescription"].values[0]).content

    return query


def simularity_search(query:str, db, chunks_number:int=4) -> str:
    docs = db.similarity_search(query, k=chunks_number)  # find chunks_number docs similar to the user's query; FAISS does the similarity search
    new_company_report_chunks_summary = " ".join([doc.page_content for doc in docs])  # combine "page_content" fields from each of the found docs
    return new_company_report_chunks_summary


def prep_prompt_inputs(new_company, metrics, metric_id, train_examples, new_company_report_chunks_summary) -> dict:
    prompt_inputs = {
        'new_company': new_company,
        'metric_id': metric_id,
        'metric_name': metrics[metrics["MetricID"]==metric_id]["MetricName"].values[0],
        'metric_description': metrics[metrics["MetricID"]==metric_id]["MetricDescription"].values[0]
    }

    # add inputs from exaples
    df = train_examples[train_examples['MetricID']==metric_id].reset_index(drop=True)
    assert len(df) == 3, "Expected exactly 3 example companies for 1 metric"  #Safety check  #FIXME:if we add more trainig examples later

    prompt_inputs['Company_1'] = df.loc[0, 'Company']
    prompt_inputs['Score_1'] = df.loc[0, 'Score']
    prompt_inputs['Reason1_1'] = df.loc[0, 'Reason1']
    prompt_inputs['Reason2_1'] = df.loc[0, 'Reason2']
    prompt_inputs['Reason3_1'] = df.loc[0, 'Reason3']
    
    prompt_inputs['Company_2'] = df.loc[1, 'Company']
    prompt_inputs['Score_2'] = df.loc[1, 'Score']
    prompt_inputs['Reason1_2'] = df.loc[1, 'Reason1']
    prompt_inputs['Reason2_2'] = df.loc[1, 'Reason2']
    prompt_inputs['Reason3_2'] = df.loc[1, 'Reason3']
    
    prompt_inputs['Company_3'] = df.loc[2, 'Company']
    prompt_inputs['Score_3'] = df.loc[2, 'Score']
    prompt_inputs['Reason1_3'] = df.loc[2, 'Reason1']
    prompt_inputs['Reason2_3'] = df.loc[2, 'Reason2']
    prompt_inputs['Reason3_3'] = df.loc[2, 'Reason3']

    prompt_inputs['new_company_report_chunks_summary'] = new_company_report_chunks_summary
    
    return prompt_inputs


if __name__ == "__main__":
    AI_Provider = "GOOGLE"
    # LLM_Provider = "OPENAI"

    llm, llm_structured = get_llm(AI_Provider)
    print("My LLM version is:", llm.invoke("What LLM version are you?").content)

#     print(f'Response:\n{}')
