# match_pipeline/match_agent.py

import openai
import re

from tools import embedding_tool, compute_similarity

class ProfileMatchingAgent:
    """
    Demonstration agent that:
      1) Summarizes job description
      2) Summarizes CV
      3) Uses embeddings to calculate an initial similarity
      4) Makes a final LLM call to produce a 0-10 match rating + bullet explanation
    No conversation memory is stored across steps; each step is independent.
    """

    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    def run(self, job_text: str, cv_text: str) -> dict:
        """
        Orchestrates multi-step logic to produce a final match score & explanation.
        Returns a dict with { "score": int, "explanation": str }.
        """
        # STEP 1: Summarize Job
        job_summary = self.summarize_text(job_text, label="Job Description")

        # STEP 2: Summarize CV
        cv_summary = self.summarize_text(cv_text, label="Candidate CV")

        # STEP 3: Compute embedding similarity
        similarity_score = self.embedding_comparison(job_summary, cv_summary)

        # STEP 4: Final LLM rating
        final_result = self.final_match_rating(job_summary, cv_summary, similarity_score)
        return final_result

    def summarize_text(self, text: str, label: str) -> str:
        """
        Summarize the input text via LLM. 
        Each call is isolated; no memory usage between calls.
        """
        prompt = f"""
You are an AI assistant. Please summarize the following {label} in concise bullet points:
{text}
""" 
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        summary =  response.choices[0].message.content.strip()
        # summary = response["choices"][0]["message"]["content"].strip()
        return summary

    def embedding_comparison(self, job_summary: str, cv_summary: str) -> float:
        """
        Use embeddings on the two summaries and compute a similarity score (0-1).
        """
        vec_job = embedding_tool(job_summary)
        vec_cv = embedding_tool(cv_summary)
        return compute_similarity(vec_job, vec_cv)

    def final_match_rating(self, job_summary: str, cv_summary: str, similarity_score: float) -> dict:
        """
        Make a final LLM call. Provide:
         - match score (0-10)
         - bullet-point explanation
        No ongoing conversation state; just a direct prompt.
        """
        prompt = f"""
You are an AI specialized in ranking CVs against job descriptions.

[JOB SUMMARY]
{job_summary}

[CV SUMMARY]
{cv_summary}

Embedding similarity is {similarity_score:.3f} on a 0 to 1 scale.

Please provide:
1) A match score (0 to 10).
2) A short bullet-point explanation of strengths/weaknesses.
Format it as:
"Score: X
- Bullet point
- Bullet point"
"""
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        # content = response["choices"][0]["message"]["content"]
        match_score = 0

        # Attempt to parse out the score from the response
        score_pattern = re.compile(r"Score:\s*(\d{1,2})")
        found = score_pattern.search(content)
        if found:
            match_score = int(found.group(1))

        return {
            "score": match_score,
            "explanation": content
        }