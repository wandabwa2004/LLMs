
import openai
import os
from openai import OpenAI  # Import the OpenAI class

def build_prompt(user_question, retrieved_chunks):
    prompt = "You are a helpful chatbot designed to answer questions about Fitness Passport.\n\n"
    prompt += "Here is some information from our FAQs that might be relevant to the user's question:\n---\n"
    for chunk in retrieved_chunks:
        question, answer = chunk.split('\n', 1)
        prompt += f"Question: {question}\nAnswer: {answer}\n---\n"
    
    prompt += f"\nUser Question: {user_question}\n\n"
    prompt += "Based on the information provided above, answer the user's question to the best of your ability. If the context does not answer the question, you can tell the user that you don't have the answer."
    return prompt

def format_response_with_references(response_text, retrieved_metadatas): # we create a function for the references
    """
    Formats the response text with clickable references to the source documents.

    Args:
        response_text: The chatbot's response text.
        retrieved_metadatas: A list of metadata dictionaries returned from ChromaDB.

    Returns:
        The formatted response text with clickable references.
    """
    formatted_response = response_text + "\n\n" # we add the original response
    formatted_response += "References:\n"
    for metadata in retrieved_metadatas: # we loop in the metadatas
        source = metadata["source"]
        formatted_response += f"- [{source}]({source})\n" # we create the links using the source text
    return formatted_response

def get_openai_response(prompt, retrieved_metadatas): # we now pass the retrieved_metadatas
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI()  # Create an instance of the OpenAI client
    response = client.chat.completions.create(  # Use the new API interface
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    response_text=response.choices[0].message.content
    formatted_response = format_response_with_references(response_text, retrieved_metadatas) # we use our function to create the references
    return formatted_response
