import os
import textwrap
import streamlit as st
import asyncio
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from llama_parse import LlamaParse
import nltk
import time

# Load environment variables
load_dotenv()

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Set environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Function to print response
def print_response(response):
    response_txt = response["result"]
    for chunk in response_txt.split("\n"):
        if not chunk:
            st.write()
            continue
        st.write("\n".join(textwrap.wrap(chunk, 100, break_long_words=False)))

# Instructions for parsing
instruction = """The provided document is the Constitution of Kenya 2010. This document encompasses all the legal frameworks, 
                  guidelines, and principles governing the country. It defines the structure of the state, the distribution of 
                  powers between different levels of government, and the fundamental rights and duties of citizens.

When answering questions based on this document, please follow these guidelines:
- Be precise and concise in your responses.
- Ensure that the information is accurate and directly relevant to the question.
- Highlight key articles, sections, or provisions where applicable.
- Provide context to your answers when necessary, explaining the implications or importance of specific provisions.
- Maintain a neutral and informative tone, avoiding any personal opinions or interpretations.

The goal is to provide clear and informative answers that help the user understand the specific aspects of the Constitution of Kenya 2010."""

# Initialize LlamaParse
parser = LlamaParse(
    api_key=os.getenv("LLAMA_PARSE_API_KEY"),
    result_type='markdown',
    parsing_instruction=instruction,
    max_timeout=5000
)

# Load and parse the document
@st.cache_resource
def load_and_parse_document(file_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    llama_parse_document = loop.run_until_complete(parser.aload_data(file_path))
    parsed_doc = llama_parse_document[0]
    return parsed_doc

# Save parsed document as Markdown
def save_parsed_document(parsed_doc, output_path):
    with open(output_path, "w") as f:
        f.write(parsed_doc.text)

# Load and split documents
@st.cache_resource
def load_and_split_documents(document_path):
    loader = UnstructuredMarkdownLoader(document_path)
    loaded_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
    docs = text_splitter.split_documents(loaded_documents)
    return docs

# Create and load vector store
@st.cache_resource
def create_vector_store(_docs, _embeddings, path, collection_name):
    qdrant = Qdrant.from_documents(_docs, _embeddings, path=path, collection_name=collection_name)
    return qdrant

# Initialize embeddings
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Streamlit UI with Icons and Descriptions
st.title("📜 Kenyan Constitution Chatbot 🇰🇪")
st.write("### 🧾 Understand and Query the **Constitution of Kenya 2010**")
st.markdown(
    """
    This application allows you to upload the Constitution of Kenya 2010 (PDF), parse it, and retrieve concise answers to your questions.  
    **Features include:**
    - 🛠️ Document Parsing
    - 🧩 Document Splitting for Analysis
    - 🔍 Intelligent Search using Vector Stores
    - 🤖 AI-Generated Responses (powered by Groq and LlamaParse)
    """
)

uploaded_file = st.file_uploader("📂 **Upload the Constitution of Kenya 2010 PDF**", type=["pdf"])
if uploaded_file is not None:
    st.info("📑 **Document Uploaded Successfully! Processing...**")
    file_path = "/tmp/uploaded_file.pdf"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    parsed_doc = load_and_parse_document(file_path)
    st.success("✅ Document Parsing Complete!")
    # st.write("📝 **Parsed Document (Preview):**")
    # st.code(parsed_doc.text[:1000], language="markdown")

    output_path = "/tmp/parsed_const_document.md"
    save_parsed_document(parsed_doc, output_path)

    docs = load_and_split_documents(output_path)
    st.success("🔍 **Document Splitting Complete! Ready for Querying.**")

    qdrant = create_vector_store(docs, embeddings, path="/tmp/qdrant_db", collection_name="document_embeddings")
    st.success("📊 **Vector Store Created! Start asking questions.**")

    query = st.text_input("💬 **Enter your question:**")
    if st.button("🚀 Submit"):
        with st.spinner("🔎 **Searching for the most relevant answers...**"):
            retriever = qdrant.as_retriever(search_kwargs={"k": 5})
            retrieved_docs = retriever.invoke(query)

            compressor = FlashrankRerank(model='ms-marco-MiniLM-L-12-v2')
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever
            )

            reranked_docs = compression_retriever.invoke(query)

            llm = ChatGroq(temperature=0, model='llama3-70b-8192')

            prompt_template = """
            Use the following pieces of information to answer the user's question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Context: {context}
            Question: {question}

            Answer the question and provide additional helpful information,
            based on the pieces of information, if applicable. Be succinct.

            Responses should be properly formatted to be easily read.
            """

            prompt = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=compression_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt, "verbose": True},
            )

            start_time = time.time()
            response = qa.invoke(query)
            end_time = time.time()
            st.success(f"⏱️ **Response Time: {end_time - start_time:.2f} seconds**")

            st.subheader("📋 **Answer:**")
            print_response(response)