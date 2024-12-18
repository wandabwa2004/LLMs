import openai
from document_processor import index, doc_store
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def set_openai_api_key(api_key):
    """Sets the OpenAI API key dynamically."""
    openai.api_key = api_key

def is_similar(text1, text2, threshold=0.85):
    """Check if two texts are similar using SequenceMatcher."""
    return SequenceMatcher(None, text1, text2).ratio() > threshold

def filter_duplicates(sources, threshold=0.85):
    """Filter out duplicate or overly similar chunks."""
    unique_sources = []
    for source in sources:
        if all(not is_similar(source, existing, threshold) for existing in unique_sources):
            unique_sources.append(source)
    return unique_sources

def retrieve_and_answer(query):
    # Check if API key is set
    if not openai.api_key:
        raise ValueError("OpenAI API key not set. Please provide a valid API key.")

    # Embed the query
    query_embedding = embedding_model.encode([query])
    
    # Retrieve relevant chunks
    distances, indices = index.search(query_embedding, k=10)  # Retrieve more to allow filtering
    sources = [doc_store[i] for i in indices[0]]
    
    # Filter out duplicate or overly similar sources
    sources = filter_duplicates(sources)
    
    # Limit to top 3 unique sources for context
    context = "\n".join(sources[:3])
    
    # Use LLM to generate an answer
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Use "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer the following question based on the context provided:\n\nContext:\n{context}\n\nQuestion:\n{query}"}
        ],
        max_tokens=200,
        temperature=0.7
    )
    answer = response.choices[0].message.content.strip()
    
    return answer, sources[:5]  # Return top 5 sources (filtered and unique)