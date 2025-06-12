import streamlit as st
import langchain_helper as lch
import textwrap

st.title("YouTube Assistant")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(
            label="Input YouTube URL:",
            max_chars=50
            )  # "https://www.youtube.com/watch?v=5MgBikgcWnY&cc_load_policy=1&hl=es"
        query = st.sidebar.text_area(
            label="Input query:",
            max_chars=50,
            key="query"
            )  # "What is the main idea of this video?"
        gemini_api_key = st.sidebar.text_input(
            label="Input Google Gemini API key:",
            key="langchain_search_api_key_openai",
            max_chars=50,
            type="password"
            )
        "[Get an OpenAI API key](https://aistudio.google.com/app/apikey)"
        "[View the source code](https://github.com/AndrianovAL/LangChain_RAG_YouTube/tree/main)"
        submit_button = st.form_submit_button(label='Submit')

if youtube_url and query:
    if not gemini_api_key:
        st.info("Please add your Google Gemini API key to continue.")
        st.stop()
    else:
        db = lch.make_vectordb_from_youtube_url(youtube_url)
        if db:
            response, docs = lch.get_response_from_query(db, query, gemini_api_key)
            st.subheader("Answer:")
            st.text(textwrap.fill(response, width=85))