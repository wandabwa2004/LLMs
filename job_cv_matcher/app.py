# app.py
import streamlit as st
import openai
from text_extraction import extract_text
from text_cleaning import clean_text
from match_agent import ProfileMatchingAgent

def main():
    # Title & Basic Description
    st.set_page_config(page_title="Job/CV Matching System", page_icon=":briefcase:")
    st.title("Job/CV Matching System :mag_right:")

    # Info about the App
    st.markdown(
        """
        **Welcome!** This application helps you match multiple CVs against a given job description 
        by leveraging a multi-step AI model (an LLM).  
        
        **Key features**:
        - **Text or File Upload** for the job description.
        - **Multiple CV Uploads** (PDF, DOCX, or TXT).
        - **AI-Based Ranking** of CVs based on how well they match the job.
        - **Bullet-Point Explanation** to highlight strengths and weaknesses of each CV.

        **How it works** (under the hood):
        1. We summarize the job description with the AI.  
        2. We summarize each CV.  
        3. We compute a vector similarity (embeddings) between the summaries.  
        4. We call the AI again to produce a final **0-10 match score** and an explanation.

        ---
        """
    )

    # Optionally let user enter their OpenAI API key (🔑 icon)
    st.subheader(":key: OpenAI API Key")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if openai_api_key:
        openai.api_key = openai_api_key

    # Let user choose how to provide the Job Description
    st.subheader(":clipboard: Job Description Input Method")
    job_input_method = st.radio(
        "Select how to provide the job description",
        options=["Text Input", "File Upload"]
    )

    # Variables to store final job description text
    job_text_clean = ""

    if job_input_method == "Text Input":
        # Provide a text area to copy-paste job description
        st.markdown(":pencil2: **Paste the Job Description Below:**")
        job_desc_text = st.text_area("", height=200)
        if job_desc_text:
            job_text_clean = clean_text(job_desc_text)
    else:
        # "File Upload"
        st.markdown(":open_file_folder: **Upload a Job Description File (PDF, DOCX, or TXT):**")
        job_description_file = st.file_uploader(
            label="",
            type=["pdf", "docx", "txt"]
        )
        if job_description_file:
            job_text_raw = extract_text(job_description_file)
            job_text_clean = clean_text(job_text_raw)

    # Upload CVs
    st.subheader(":file_folder: Candidate CV Uploads")
    cv_files = st.file_uploader(
        "Upload one or more CVs (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    # Button to Match
    st.markdown("---")
    if st.button(":rocket: Match CVs to Job Description"):
        # Check if API key is set
        if not openai.api_key:
            st.warning("Please provide your OpenAI API key above.")
            return

        # Check if we have job text and at least one CV
        if job_text_clean and cv_files:
            # Create the Agent (no memory)
            agent = ProfileMatchingAgent(model="gpt-3.5-turbo")

            results = []
            for cv_file in cv_files:
                cv_text_raw = extract_text(cv_file)
                cv_text_clean = clean_text(cv_text_raw)

                # Run the multi-step agent logic
                match_result = agent.run(job_text_clean, cv_text_clean)

                results.append({
                    "file_name": cv_file.name,
                    "score": match_result["score"],
                    "explanation": match_result["explanation"]
                })

            # Sort by score descending
            results.sort(key=lambda x: x["score"], reverse=True)

            st.subheader(":trophy: Match Results (Highest to Lowest)")
            for i, r in enumerate(results, start=1):
                st.markdown(f"**Rank {i}**: `{r['file_name']}` | **Score:** {r['score']}")
                st.write("**Explanation / Summary:**")
                st.write(r["explanation"])

        else:
            st.warning("Please provide both a job description (via text or file) and at least one CV.")

if __name__ == "__main__":
    main()

