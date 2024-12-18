A secure, privacy-focused Retrieval-Augmented Generation (RAG) system designed for local document-based Question-Answering (QA). This application enables users to upload documents, extract relevant text, and retrieve answers to queries using OpenAI's GPT models—all while keeping data local.

# Features:

 **Document Upload:**
 
   -  Upload and process documents in supported formats (PDFs, Word files, and plain text files).

**Text Extraction:**

  - Automatically extract and preprocess text from uploaded documents.

**Vector Store for Retrieval:**

  - Stores document content as embeddings and retrieves the most relevant sections for user queries.

**Interactive Q&A:**

  - Provides accurate answers to user queries based on document context.

**Streamlit Interface:**
  - A user-friendly interface for managing documents and querying.

**Local Processing:**
  - Ensures maximum privacy by processing data locally.

# Tech Stack

1. OpenAI API: For generating accurate and context-aware answers.
2. LangChain: For embedding and retrieval pipelines.
3. FAISS: A local vector database for efficient similarity search.
4. Streamlit: A Python-based framework for building the user interface.
5. Python: Backend logic and integration.

# File Structure

RAG_chat_openAI/
├── app.py               # Main Streamlit app for the interface
├── document_processor.py # Functions for document upload and text extraction
├── question_answering.py # Handles LLM-based question answering
├── vector_store.py       # Embedding and vector database management
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

# How It Works

**Document Upload:**
 - Users upload documents via the Streamlit interface.
 - Supported formats include PDFs, Word files, and plain text files.

**Text Processing:**
 - Uploaded documents are processed by document_processor.py to extract raw text.

**Vector Embedding and Storage:**
 - The text is embedded using a language model and stored locally in FAISS (via vector_store.py).

**Question Answering:**
 - User queries are processed by question_answering.py, which retrieves relevant document sections from the vector store.
 - Retrieved sections are used as context for generating answers with an LLM (e.g., GPT-4).

**User Interaction:**
 - The system displays answers alongside source document sections for transparency.

# Installation and Setup

**Prerequisites**
- Python 3.8 or later
- An OpenAI API key

## Installation

1. Clone the repository:
```git clone https://github.com/wandabwa2004/RAG_chat_openAI.git```
```cd RAG_chat_openAI```
2. Install dependencies:
```pip install -r requirements.txt```
3. Set up the OpenAI API key:
  Create a .env file in the project root and add your API key:
  ```OPENAI_API_KEY=your_openai_api_key```
4. Run the Streamlit app:
 ```streamlit run app.py```
5. Open your browser and navigate to the URL displayed (usually http://localhost:8501).
