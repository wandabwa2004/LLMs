# Safaricom Smart Assistant: A RAG System with MMR

This repository contains a Retrieval-Augmented Generation (RAG) system implementation tailored for Safaricom—a Telkom company in Kenya known for its MPESA service. The project combines web scraping, embedding generation, Maximum Marginal Relevance (MMR) for re-ranking, and response generation via OpenAI's GPT model. A Gradio-based interface allows users to interact with the smart assistant in real time.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Scraping FAQs](#scraping-faqs)
  - [Launching the Smart Assistant](#launching-the-smart-assistant)
- [Deployment Considerations](#deployment-considerations)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a RAG system that enhances customer support by combining:
- **Data Scraping:** Extracts FAQs and answers from Safaricom's FAQ page using Python's `requests` and `BeautifulSoup`.
- **Response Generation:** Leverages a combination of FAISS indexing, SentenceTransformer embeddings, and MMR-based re-ranking to retrieve relevant FAQ entries. The system then uses OpenAI's GPT model to generate context-aware responses.
- **Interactive Interface:** A user-friendly Gradio interface for querying the system.

---

## Features

- **Web Scraping:** Robust extraction of FAQ pairs with random delays to mimic human behavior.
- **Embedding & Indexing:** Uses SentenceTransformer and FAISS to build and query an efficient embedding index.
- **MMR Reranking:** Implements Maximum Marginal Relevance to ensure response diversity and relevance.
- **Response Generation:** Integrates OpenAI's GPT-3.5-turbo to generate structured responses.
- **Interactive UI:** Gradio-based interface for seamless user interactions.

---

## Project Structure

```
├── app.py                # Main application with Gradio interface and response generation logic.
├── scraper.py            # Script for scraping Safaricom FAQ pages and saving FAQ pairs.
├── faq_data.json         # JSON file containing the scraped FAQ data (generated after running scraper.py).
├── requirements.txt      # List of required Python packages.
└── README.md             # This file.
```
## Installation
1. Clone the Repository:

```
git clone https://github.com/yourusername/safaricom-smart-assistant.git
cd safaricom-smart-assistant
```
2. Create a Virtual Environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install Dependencies:

`pip install -r requirements.txt`

## Usage

### Scraping FAQs
Before running the smart assistant, scrape the FAQs from Safaricom's website:

`python scraper.py`

This will create a file named `faq_data.json` containing all the scraped FAQ pairs.

### Launching the Smart Assistant

After generating the FAQ data, run the Gradio interface:

`python app.py`

The Gradio interface will launch in your browser. Enter your OpenAI API key and a query (e.g., "How do I register for MPESA?") to interact with the smart assistant.

### Deployment Considerations

**Local Deployment:**

Use the steps above to run the application locally for testing and development.

**Cloud Deployment:**

- Containerize the application using Docker for consistent environments.
- Deploy on cloud platforms such as AWS, Azure, or GCP.
- Secure your OpenAI API key using environment variables or cloud secrets management.
- Consider using persistent storage for the FAISS index and embeddings.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have suggestions, bug fixes, or improvements.

## License
This project is licensed under the MIT License.
