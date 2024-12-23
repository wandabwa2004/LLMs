# match_pipeline/tools.py

import openai
import numpy as np

def embedding_tool(text: str, model="text-embedding-ada-002"):
    """
    Calls the OpenAI Embeddings API to get a vector for the input text.
    """
    if not text.strip():
        return None

    response = openai.embeddings.create(
        model=model,
        input=text
    )
    # vector = response["data"][0]["embedding"]
    vector  = response.data[0].embedding

    return np.array(vector)

def compute_similarity(vector_a, vector_b):
    """
    Cosine similarity for two numpy arrays.
    """
    if vector_a is None or vector_b is None:
        return 0
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if (norm_a * norm_b) == 0:
        return 0
    return dot_product / (norm_a * norm_b)