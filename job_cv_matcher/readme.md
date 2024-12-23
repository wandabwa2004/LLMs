# Job/CV Matching System

A multi-step AI application that matches one or more CVs against a Job Description using an LLM-based approach. Users can copy-paste or upload a job description (PDF, DOCX, or TXT), then upload multiple CVs, and the system will rank them based on 0–10 match scores and provide explanations of the strengths and weaknesses for each CV.

## Features
- **Flexible Job Description Input**
Either copy-paste directly into a text area or upload a file in PDF, DOCX, or TXT.

- **Multiple CV Uploads**
Drag and drop (or browse) multiple CV files of various formats.

- **AI-Generated Matching & Explanation**
The system uses a multi-step logic:
1. Summarize the job description.
2. Summarize each CV.
3. Compute an embedding-based similarity.
4. Generate a final match score (0–10) and bullet-point explanation.

- **Interactive Web Interface**
Built with Streamlit, ensuring a simple, browser-based UI.

## Technologies

- Python 3.9+ (recommended)
- Streamlit for the web interface
- OpenAI (GPT-3.5 or GPT-4) for LLM calls
- pdfplumber, docx2txt for file parsing
- Numpy for embedding-based similarity

## Project Structure
my_job_matcher/
├── crew.yml                 # (Optional) CrewAI orchestration file
├── requirements.txt         # Python dependencies
├── app.py                   # Streamlit application
├── init__.py
├── text_extraction.py   # Functions to parse PDF, DOCX, or TXT
├── text_cleaning.py     # Basic text cleaning
├── tools.py             # Embedding + similarity calculations
└── match_agent.py       # Multi-step agent logic (LLM calls)

```app.py```: Main Streamlit app. This is the entry point for the user interface.

## Installation
1. Clone the Repository:
```
git clone https://github.com/YourUsername/my_job_matcher.git
cd my_job_matcher
```
2. Install Dependencies:
```pip install -r requirements.txt```
3. Set Your OpenAI API Key:
   - Export it as an environment variable:
     ```export OPENAI_API_KEY="sk-YOUR_OPENAI_KEY"```
     or
   - Provide it inside the Streamlit app (there’s a text field to enter the API key).
  
## Usage
1. **Run the App:**
```streamlit run app.py```
The terminal should display a local URL, such as http://localhost:8501. The port number may be different on your ststem  Open that in your browser.

2. **Provide the Job Description:**

- Use Text Input to paste the job description, or
- Use File Upload to upload a PDF, DOCX, or TXT file.

3. **Upload CVs:**

- Click the CV uploader to select one or more CV files in PDF, DOCX, or TXT.

4. **Click “Match CVs to Job Description”:**

- The system will summarize the job, summarize each CV, compute embedding similarity, and call the LLM for a match score.
- Results are shown in descending order (highest to lowest score).

**Example Walkthrough**
1. **API Key**: Enter your OpenAI API Key in the dedicated field.
2. **Job Input**: Choose “File Upload” → select a Software_Engineer_Job.pdf file.
3. **CV Uploads**: Upload multiple CVs, e.g., Candidate1.docx, Candidate2.pdf.
4. **Run**: Click the “Match CVs to Job Description” button.
5. **Results**: See a table of ranks, scores, and bullet-point explanations for each CV.

Contact
Project Maintainer: Herman Wandabwa
Email: wandabwa2004@gmail.com
GitHub: @wandabwa2004

_Thank you for using the Job/CV Matching System! We hope it streamlines your recruitment workflow with AI-driven insights._
