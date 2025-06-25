from langchain_google_genai import GoogleGenerativeAI  # LLM: Google Gemini
    # Maybe need:   from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate  # for templates
from langchain.chains import LLMChain  # chain llm & prompt


def get_response_from_query(db, query, api_key=None, k=4):
    ''' Want to constrain the number of tokens sent to 4000 '''
    docs = db.similarity_search(query, k=k)  # find 4 docs similar to the user's query; FAISS does the similarity search
    docs_page_content = " ".join([doc.page_content for doc in docs])  # combine "page_content" fields from each of the found docs

    if api_key == None:
        load_dotenv()
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    else:
        GOOGLE_API_KEY = api_key
    llm = GoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-2.0-flash", temperature=0.2, max_tokens=6_000)
        # Maybe need:   llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-2.0-flash", temperature=0.2, max_tokens=500)
        # Alternative:  llm = OpenAI(model_name="text-davinci-003")
 
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos based on the video's transcript.

        Answer the following question: {question}
        By searching the following video transcript: {docs}

        Only use the factual information from the transcript to answer the question.
        If you feel like you don't have enough information to answer the question, say "I don't know".
        Your answers should be verbose and detailed, but limited to 1500 tokens.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


if __name__ == "__main__":
    report_filename = 'data\data\Metrics.csv'
    db = make_vectordb_from_report(report_filename)
    query = "What company is this report for?"

    response, docs = get_response_from_query(db, query)

    print(f'Response:\n{response}')
