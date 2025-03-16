import streamlit as st
import os
from data_handler import DataHandler
from utils import build_prompt, get_openai_response
import time

# Data loading and setup
data_path = "output/faqs.json"  # Replace with your actual path
persist_directory = "./chroma_db"
# Instantiate DataHandler without tenant/database parameters
data_handler = DataHandler(data_path, persist_directory=persist_directory)

# Initialize collection in session state if it doesn't exist
if "collection" not in st.session_state:
    # Check if the collection exists. If not, create it
    try:
        data_handler.chroma_client.get_collection(name=data_handler.collection_name)  # we try to get it
        print("Collection exists")
        with st.spinner("Deleting old collection"):
            data_handler.delete_chroma_collection()  # we delete it
            time.sleep(1)  # we wait
        print("Collection deleted")
        print(f"Collection does not exist: we need to create it")
        with st.spinner("Processing data and creating collection..."):
            collection = data_handler.process_data_and_create_collection()  # we create the collection
        st.session_state["collection"] = collection

    except Exception as e:  # if we cant, an error is raised
        print(f"Collection does not exist: {e}. We need to create it")
        with st.spinner("Processing data and creating collection..."):
            collection = data_handler.process_data_and_create_collection()  # we create the collection
        st.session_state["collection"] = collection

# Streamlit app
st.title("Fitness Passport Customer Support Chatbot")

# API Key Input
st.sidebar.header("API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# Set the api key for the session if the user provides it
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.sidebar.success("API key set!")

# Check if the API key is set
if not os.environ.get("OPENAI_API_KEY"):
    st.error("Please enter your OpenAI API key in the sidebar.")
    st.stop()  # Stop the app if no API key

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG flow
    with st.spinner("Getting the relevant documents..."):
        results = data_handler.query_chroma(prompt)  # we get the full results (documents and metadatas)
        if results["documents"] and results["metadatas"]:  # if there are documents and metadatas
            retrieved_chunks = results["documents"][0]
            retrieved_metadatas = results["metadatas"][0]  # we get the metadatas
        else:
            retrieved_chunks = []
            retrieved_metadatas = []

    full_prompt = build_prompt(prompt, retrieved_chunks)
    with st.spinner("Generating the response..."):
        response = get_openai_response(full_prompt, retrieved_metadatas)  # we add the metadatas

    # Streamlit display
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)  # we now display the response in markdown
