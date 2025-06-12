# %pip install -U langchain-huggingface
from langchain_community.document_loaders import YoutubeLoader  # to load YouTube transcript
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound  # to handle exceptions if can't load a transcript
from xml.etree.ElementTree import ParseError  # to handle exceptions if can't load a transcript
from langchain.text_splitter import RecursiveCharacterTextSplitter  # to split YouTube transcript into chunks
from langchain_huggingface import HuggingFaceEmbeddings
    # Old:          from langchain.embeddings import HuggingFaceEmbeddings
    # Alternative:  from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS  # Vector Database (indexes); alternatives: Pinecon, Weaviate
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAI  # LLM: Google Gemini
    # Maybe need:   from langchain_google_genai import ChatGoogleGenerativeAI
    # Alternative:  from langchain.llms import OpenAI  # LLM: OpenAI
from langchain.prompts import PromptTemplate  # for templates
from langchain.chains import LLMChain  # chain llm & prompt


def safe_load_transcript(video_url):
    try:
        loader = YoutubeLoader.from_youtube_url(video_url)  # initialize the loader with my URL
        transcript = loader.load()  # use the loader to load the transcript
        return transcript
    except (TranscriptsDisabled, NoTranscriptFound, ParseError) as e:
        print(f"[WARN] Transcript not available or failed to parse: {e}")
        return None

def make_vectordb_from_youtube_url(video_url: str) -> FAISS:
    transcript = safe_load_transcript(video_url)
    if transcript:
        print("Transcript loaded.")
    else:
        print("Transcript could not be loaded.")
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # initilize the text splitter
    docs = text_splitter.split_documents(transcript)  # split my transcript into overlapping chunks
    # print(f'Test: docs[0].page_content:\n{docs[0].page_content}')  # test

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # This model is small (~80MB), fast on CPU, good for English YouTube transcripts
        # Alternative: ultra-fast memory-light (~45MB): model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
        # Alternative: embeddings = OpenAIEmbeddings()
    # # Test: one embed string
    # embed_test = embeddings.embed_query("What is the main idea of this video?")
    # print(f'Test: len(embeddings)={len(embed_test)}, embeddings[:5]={embed_test[:5]}')  # Should return a 384-dim vector
    db = FAISS.from_documents(docs, embeddings)  # create a DB of vector embeddings from the docs
    return db

def get_response_from_query(db, query, api_key=None, k=4):
    ''' Want to constrain the number of tokens sent to 4000 '''
    docs = db.similarity_search(query, k=k)  # find 4 docs similar to the user's query; FAISS does the similarity search
    docs_page_content = " ".join([doc.page_content for doc in docs])  # combine "page_content" fields from each of the found docs

    if api_key == None:
        load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    else:
        GEMINI_API_KEY = api_key
    llm = GoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model="gemini-2.0-flash", temperature=0.2, max_tokens=6_000)
        # Maybe need:   llm = ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model="gemini-2.0-flash", temperature=0.2, max_tokens=500)
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
    video_url = "https://www.youtube.com/watch?v=5MgBikgcWnY"
    query = "What is the main idea of this video?"

    db = make_vectordb_from_youtube_url(video_url)
    if db:
        response, docs = get_response_from_query(db, query)
        print(f'Response:\n{response}')
