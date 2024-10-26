import requests
import streamlit as st
from transformers import pipeline
import os

qa_pipeline = pipeline("question-answering")

def perform_web_search(query):
    """
    Performs a web search using Bing Search API and retrieves results.
    """
    try:
        search_url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": os.getenv("BING_SEARCH_API_KEY")}
        response = requests.get(search_url, headers=headers, params={"q": query})
        response.raise_for_status()
        return response.json().get("webPages", {}).get("value", [])
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP error occurred: {e}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error during the request: {e}")
    return []

def extract_answer_from_context(context, question):
    """
    Uses the QA pipeline to extract an answer based on context.
    """
    return qa_pipeline(question=question, context=context)["answer"]

st.set_page_config(page_title="Web Search and QA", page_icon="🔍")
st.title("Web Search and Question Answering")

search_query = st.text_input("Enter your search query:")

if st.button("Search"):
    if search_query:
        search_results = perform_web_search(search_query)
        if search_results:
            st.success("Search completed!")
            for result in search_results:
                st.write(f"**Title:** {result['name']}")
                st.write(f"**Link:** [Visit]({result['url']})")
                st.write(f"**Snippet:** {result['snippet']}\n")
        else:
            st.error("No results found.")
    else:
        st.warning("Please enter a search query.")

user_question = st.text_input("Ask a question based on the search results:")

if user_question and search_results:
    context = " ".join([result['snippet'] for result in search_results])
    answer = extract_answer_from_context(context, user_question)
    st.write("### Answer:")
    st.write(answer if answer else "No answer found.")
