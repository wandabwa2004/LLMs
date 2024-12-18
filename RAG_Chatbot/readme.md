# 📜 Kenyan Constitution Chatbot 🇰🇪

## Description  
The **Kenyan Constitution Chatbot** is an AI-powered application that allows users to upload the **Constitution of Kenya 2010** PDF file, parse it, and ask specific questions about its content. It leverages **AI models** and **document embeddings** to provide concise, accurate, and context-aware answers.

---

## Features  

- 🛠️ **Document Parsing**: Converts uploaded PDFs into machine-readable text.  
- 🔍 **Intelligent Search**: Uses vector search and document splitting to ensure efficient querying.  
- 📊 **AI-Generated Answers**: Provides precise, formatted answers to questions based on the document.  
- 🚀 **Fast Processing**: Optimized response time for queries.  
- 💬 **Interactive Chat**: Users can input queries and receive context-rich answers.  

---

## Technology Stack  

- **Frontend**: Streamlit  
- **Document Parsing**: LlamaParse API  
- **Vector Store**: Qdrant  
- **Embeddings Model**: FastEmbed (BAAI/bge-base-en-v1.5)  
- **Language Model**: Groq API (llama3-70b-8192)  
- **Document Loader**: UnstructuredMarkdownLoader  
- **Compression**: Flashrank Rerank  
- **Environment Setup**: Python, NLTK, dotenv  

---

## Prerequisites  

Before running this application, ensure you have the following:  

1. **Python 3.9 or higher**  
2. **Required API Keys**:  
   - `GROQ_API_KEY` (Groq API Key)  
   - `LLAMA_PARSE_API_KEY` (LlamaParse API Key)  
3. **Dependencies** installed:  
   ```bash
   pip install streamlit langchain langchain_groq qdrant-client llama_parse unstructured 
   pip install langchain_community flashrank fastembed python-dotenv nltk

## Installation and Setup  

Follow these steps to install and set up the Kenyan Constitution Chatbot:

### 1. Clone the Repository  
Clone the GitHub repository to your local machine:
```
git clone https://github.com/yourusername/kenyan-constitution-chatbot.git
cd kenyan-constitution-chatbot
```
### 2. Create a Virtual Environment (Optional but Recommended)
Set up a Python virtual environment to isolate dependencies:
```
python -m venv venv
```
#### Activate the virtual environment:
For macOS/Linux:
source venv/bin/activate
For Windows:
venv\Scripts\activate

### 3. Install Dependencies
Install all required dependencies using `pip`:
`pip install -r requirements.txt`

### 4. Set Environment Variables
Create a `.env` file in the root directory and add your API keys:
```
GROQ_API_KEY=your_groq_api_key
LLAMA_PARSE_API_KEY=your_llama_parse_api_key
```
Replace your_groq_api_key and your_llama_parse_api_key with your actual API keys.

### 5. Run the Application

Start the Streamlit application:
`streamlit run app.py`

This will launch the app in your default web browser. You can now upload the Constitution of Kenya 2010 and start interacting with the chatbot. Remember this can be  adapted to any other document. Just change the system and user prompts.

