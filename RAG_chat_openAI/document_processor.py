import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(384)  # FAISS Index
doc_store = []

def deduplicate_chunks(chunks):
    unique_chunks = list(set(chunks))  # Removes exact duplicates
    return unique_chunks

def reset_store():
    """Clears the FAISS index and document store."""
    global index, doc_store
    index = faiss.IndexFlatL2(384)  # Reinitialize the FAISS index
    doc_store = []  # Clear the document store


def process_uploaded_files(uploaded_file):
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract text from the PDF
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    
    # Chunk the text
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    chunks = deduplicate_chunks(chunks)  # Remove duplicates
    
    # Embed and store in FAISS
    embeddings = embedding_model.encode(chunks)
    index.add(embeddings)
    doc_store.extend(chunks)
