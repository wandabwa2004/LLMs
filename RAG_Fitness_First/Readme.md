# Fitness Passport RAG Chatbot

A Retrieval-Augmented Generation (RAG) system that powers a customer support chatbot for Fitness Passport. This project scrapes FAQ data from a Freshdesk support page, processes and embeds the data using Sentence Transformers, stores it in ChromaDB, and finally leverages OpenAI's GPT model to generate context-aware responses via a Streamlit interface.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Scraping FAQs](#scraping-faqs)
  - [Running the Chatbot](#running-the-chatbot)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository demonstrates a complete RAG implementation:
- **Data Scraping**: Extracts FAQs using Python's `requests` and `BeautifulSoup`.
- **Data Handling**: Cleans, chunks, and embeds the scraped data with Sentence Transformers.
- **Retrieval & Query Processing**: Stores embeddings in ChromaDB and retrieves context for user queries.
- **Response Generation**: Constructs a contextual prompt for OpenAI’s GPT and generates answers.
- **Chatbot Interface**: Provides a user-friendly interface using Streamlit.

## Features

- **Modular Codebase**: Separate modules for scraping, data handling, query processing, and UI.
- **Robust Error Handling**: Logs and handles errors gracefully during data extraction.
- **Persistent Storage**: Uses ChromaDB to persist embeddings and document collections.
- **Real-time Interaction**: Dynamic chatbot interface for real-time query response generation.

## Project Structure

├── `app.py` # Main Streamlit application integrating the RAG flow. <br>
├── `data_handler.py` # Handles data loading, chunking, embedding generation, and ChromaDB integration. <br>
├── `scraper.py` # Scrapes FAQ data from the Freshdesk support page. <br>
├── `utils.py` # Contains functions for prompt building and OpenAI API integration. <br>
├── output/ # Directory where scraped FAQs are stored (faqs.txt and faqs.json). <br>
├── chroma_db/ # Directory used by ChromaDB for persistent storage. <br>
├── `requirements.txt` # Required Python packages. <br>
└── `README.md` # This file.<br>



## Installation

1. **Clone the Repository:**

   ```
   git clone https://github.com/wandabwa2004/LLMs.git
   cd RAG_Fitness_First
   ```

2. **Create a Virtual Environment:**
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install the Dependencies:
```pip install -r requirements.txt```

## Usage
1. Scraping FAQs
To scrape FAQs from the Freshdesk support page and save them to disk, run:
```python scraper.py```
The scraped FAQs will be saved in both output/faqs.txt and output/faqs.json.

2. Running the Chatbot
To launch the chatbot application:

Make sure your scraped data is available in the output folder.
Set your OpenAI API key as an environment variable or via the Streamlit sidebar.

3. Run the Streamlit app:
```streamlit run app.py```

A browser window will open displaying the chatbot interface. Enter your query, and the application will retrieve the relevant FAQ context and generate a response using OpenAI's API.

## Deployment
For local testing, the above instructions suffice. For cloud deployment, consider the following:

Containerization: Use Docker to containerize the application.
Cloud Platforms: Deploy on AWS, GCP, or Azure. Make sure to securely manage the OPENAI_API_KEY using cloud secrets management.
Persistent Storage: If deploying horizontally, use cloud-based storage for the chroma_db directory.

## Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests if you have improvements or bug fixes.

## License
This project is licensed under the MIT License.

