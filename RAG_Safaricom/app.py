import os
import json
import numpy as np
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ------------------------------
# Configuration Parameters
# ------------------------------
CONFIDENCE_THRESHOLD = 0.6    # Reduced for MMR diversity
MMR_LAMBDA = 0.7              # Balance relevance/diversity
INITIAL_CANDIDATES = 20       # Candidates for MMR reranking
FINAL_RESULTS = 5             # Final results after reranking

# ------------------------------
# Step 1: Load and Preprocess Data
# ------------------------------
with open("faq_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

faq_list = []
for url, faq_pairs in data.items():
    for faq in faq_pairs:
        faq_list.append({
            "source": url,
            "question": faq["question"],
            "answer": faq["answer"]
        })

faq_texts = [f"Q: {faq['question']}\nA: {faq['answer']}" for faq in faq_list]

# ------------------------------
# Step 2: Build the Embedding Model and FAISS Index
# ------------------------------
embedder = SentenceTransformer("all-mpnet-base-v2")
embeddings = embedder.encode(faq_texts, show_progress_bar=True)

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype(np.float32))

# ------------------------------
# Step 3: Core RAG Functions with MMR
# ------------------------------
def mmr_rerank(query_emb, candidate_embs, lambda_param=MMR_LAMBDA, final_k=FINAL_RESULTS):
    """MMR reranking implementation"""
    query_sims = np.dot(candidate_embs, query_emb.T).flatten()
    doc_sims = np.dot(candidate_embs, candidate_embs.T)
    
    selected = []
    remaining = list(range(len(candidate_embs)))
    
    while len(selected) < final_k and remaining:
        scores = []
        for i in remaining:
            if not selected:
                score = query_sims[i]
            else:
                max_sim = np.max(doc_sims[i, selected])
                score = lambda_param * query_sims[i] - (1 - lambda_param) * max_sim
            scores.append(score)
        
        best_idx = np.argmax(scores)
        selected.append(remaining.pop(best_idx))
    
    return selected

def find_related_terms(query_text):
    """Find related questions for fallback"""
    query_embedding = embedder.encode([query_text])
    faiss.normalize_L2(query_embedding)
    _, indices = index.search(query_embedding.astype(np.float32), k=3)
    
    return "\n".join(
        f"• {faq_list[i]['question']} (Source: {faq_list[i]['source']})"
        for i in indices[0]
    )

def handle_unknown(query_text):
    """Improved unknown response handling"""
    return (
        f"I'm unable to find a complete answer for '{query_text}'. Here are related topics:\n\n"
        f"{find_related_terms(query_text)}\n\n"
        "For further assistance:\n"
        "1. Visit our [contact page](https://www.safaricom.co.ke/media-center-landing/contact-us-media)\n"
        "2. Call customer care: *100#\n"
        "3. Visit a Safaricom shop"
    )

def query_rag_enhanced(query_text):
    """Main query processing with MMR"""
    # Encode query and initial search
    query_embedding = embedder.encode([query_text])
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding.astype(np.float32), INITIAL_CANDIDATES)
    
    # Rerank with MMR
    candidate_embs = embeddings[indices[0]]
    selected = mmr_rerank(query_embedding[0], candidate_embs)
    final_indices = [indices[0][i] for i in selected]
    final_scores = scores[0][selected]
    
    # Confidence check
    if np.max(final_scores) < CONFIDENCE_THRESHOLD:
        return handle_unknown(query_text)
    
    # Build context from diverse results
    context = "\n\n".join(
        f"Source: {faq_list[i]['source']}\n"
        f"Q: {faq_list[i]['question']}\n"
        f"A: {faq_list[i]['answer']}"
        for i in final_indices
    )
    
    # Enhanced prompt for complex questions
    prompt = f"""Analyze this query for multiple components and answer using the context:

    Context:
    {context}

    Question: {query_text}

    Structure your response with:
    1. Clear section headers for different topics
    2. Numbered steps where applicable
    3. Source references in [brackets]
    4. Acknowledgment of any unaddressed aspects

    If different sources conflict, note this and provide both perspectives.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"API Error: {str(e)}"

# ------------------------------
# Step 4: Gradio Interface
# ------------------------------
 
def chatbot_interface(api_key, query_text):
    if not api_key.strip() or not query_text.strip():
        return "❗ Please provide both API key and question"
    
    global client
    client = OpenAI(api_key=api_key)
    
    return query_rag_enhanced(query_text)

iface = gr.Interface(
    fn=chatbot_interface,
    inputs=[
        gr.Textbox(label="OpenAI API Key", type="password",
                 placeholder="sk-...", max_lines=1),
        gr.Textbox(label="Your Question", placeholder="Ask about MPESA, products, services...", lines=3)
    ],
    outputs=gr.Textbox(label="Response", show_copy_button=True),
    title="🤖 Safaricom Smart Assistant",
    description=(
        "AI-powered assistant for Safaricom services. Features:\n"
        "• Handles complex, multi-part questions\n"
        "• Provides source references\n"
        "• Offers alternative solutions when uncertain"
    ),
    examples=[
        ["", "How do I register for MPESA and check my balance?"],
        ["", "What's the difference between Lipa Na MPESA and Fuliza?"]
    ],
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(share=False)