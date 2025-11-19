import json
import os
from sentence_transformers import SentenceTransformer, util
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_similarity_score(answer_ref, answer_user):
    emb_ref = embed_model.encode(answer_ref, convert_to_tensor=True)
    emb_user = embed_model.encode(answer_user, convert_to_tensor=True)
    score = util.cos_sim(emb_ref, emb_user).item()
    return score


def github_llm_request(prompt: str):
    endpoint = "https://models.github.ai/inference"
    model = "openai/gpt-4o-mini"
    token = os.getenv("MODEL_GITHUB")

    if not token:
        raise RuntimeError("MODEL_GITHUB token not found in environment variables")

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )

    response = client.complete(
        messages=[SystemMessage(prompt)],
        temperature=0.5,
        top_p=0.9,
        max_tokens=150,
        model=model,
    )

    feedback = response.choices[0].message.content.strip()
    return feedback


def llm_evaluate(reference, candidate, semantic_score):
    prompt = f"""
You are an expert evaluator for job interview answers.

First metric (precomputed):
Semantic Embedding Similarity: {semantic_score:.3f}

Semantic interpretation:
- 0.85–1.0 = almost identical meaning
- 0.70–0.85 = good similarity
- 0.50–0.70 = partial similarity
- <0.50 = weak connection

Now evaluate the candidate answer based on:

1) Meaning similarity vs reference
2) Completeness
3) Structure (logic / STAR)
4) Professionalism
5) Conciseness
6) Whether the semantic score seems correct (optional)

Return STRICT JSON:

{{
  "score": "X/10",
  "semantic_similarity": {semantic_score:.3f},
  "similarity_comment": "...",
  "strengths": ["..."],
  "issues": ["..."],
  "missing_points": ["..."],
  "final_comment": "One short paragraph summary"
}}

REFERENCE ANSWER:
\"\"\"{reference}\"\"\"

CANDIDATE ANSWER:
\"\"\"{candidate}\"\"\"
"""

    raw = github_llm_request(prompt)

    try:
        data = json.loads(raw)
    except:
        data = {"error": "Invalid JSON returned", "raw": raw}

    return data


def evaluate_pair(reference_answer, user_answer):
    semantic_score = semantic_similarity_score(reference_answer, user_answer)

    print("semantic_score", semantic_score)
    llm_score = llm_evaluate(reference_answer, user_answer, semantic_score)

    return {
        "semantic_similarity": semantic_score,
        "llm_evaluation": llm_score,
    }


# Example
if __name__ == "__main__":
    reference = "I resolved a conflict by addressing the issue calmly and collaborating on a solution."
    user = "We talked openly and found a solution together."

    result = evaluate_pair(reference, user)
    print(json.dumps(result, indent=2, ensure_ascii=False))
