# match_pipeline/text_extraction.py

import streamlit as st
import pdfplumber
import docx2txt

def extract_text_from_pdf(file) -> str:
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)

def extract_text_from_docx(file) -> str:
    return docx2txt.process(file)

def extract_text_from_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")

def extract_text(file) -> str:
    """
    Unified text extraction function for PDF, DOCX, TXT.
    """
    file_name = file.name.lower()
    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file_name.endswith(".docx"):
        return extract_text_from_docx(file)
    elif file_name.endswith(".txt"):
        return extract_text_from_txt(file)
    else:
        st.warning(f"Unsupported file type: {file.name}")
        return ""
