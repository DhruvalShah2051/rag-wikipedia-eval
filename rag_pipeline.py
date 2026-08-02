"""
Core RAG pipeline: given a question, retrieve relevant chunks from Postgres
and generate a grounded answer using Groq.

This module is imported by both 4_query.py (interactive use) and
5_evaluate.py (automated evaluation), so the retrieval/generation logic
lives in one place.
"""

import psycopg2
from sentence_transformers import SentenceTransformer
from groq import Groq
from config import DB_CONFIG, EMBEDDING_MODEL_NAME, TOP_K, GROQ_API_KEY, GROQ_MODEL

# Load once at import time so repeated calls (e.g. during evaluation) are fast
_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
_groq_client = Groq(api_key=GROQ_API_KEY)


def retrieve_chunks(query, top_k=TOP_K):
    """
    Embed the query and find the top_k most similar chunks in Postgres
    using pgvector's cosine distance operator (<=>).
    """
    query_embedding = _embedding_model.encode(query).tolist()

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT source_title, chunk_text, embedding <=> %s::vector AS distance
        FROM document_chunks
        ORDER BY distance ASC
        LIMIT %s;
        """,
        (query_embedding, top_k),
    )

    results = cur.fetchall()
    cur.close()
    conn.close()

    # Lower distance = more similar. Return as list of dicts for readability.
    return [
        {"source_title": row[0], "chunk_text": row[1], "distance": row[2]}
        for row in results
    ]


def build_prompt(query, retrieved_chunks):
    """
    Context engineering step: assemble the retrieved chunks and the user's
    question into a structured prompt that instructs the model to answer
    ONLY from the provided context (reduces hallucination).
    """
    context_block = "\n\n".join(
        f"[Source: {c['source_title']}]\n{c['chunk_text']}" for c in retrieved_chunks
    )

    prompt = f"""You are a helpful assistant answering questions using ONLY the context provided below.
If the answer is not contained in the context, say "I don't have enough information to answer that."
Do not use outside knowledge.

Answer in 1-2 complete sentences. Do not respond with just a category name, label, or single term -
briefly explain the mechanism, reasoning, or "how/why" behind your answer, even if the question sounds
like it wants a short answer.

CONTEXT:
{context_block}

QUESTION:
{query}

ANSWER:"""

    return prompt


def generate_answer(query, retrieved_chunks):
    """
    Send the assembled prompt to Groq and return the generated answer.
    """
    prompt = build_prompt(query, retrieved_chunks)

    response = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # low temperature for factual, grounded answers
    )

    return response.choices[0].message.content


def answer_question(query, top_k=TOP_K, verbose=False):
    """
    Full end-to-end pipeline: retrieve -> build prompt -> generate.
    Returns both the answer and the retrieved chunks (useful for evaluation).
    """
    retrieved = retrieve_chunks(query, top_k=top_k)
    answer = generate_answer(query, retrieved)

    if verbose:
        print(f"\nQuery: {query}")
        print("\nRetrieved chunks:")
        for c in retrieved:
            print(f"  - [{c['source_title']}] (distance={c['distance']:.4f}) {c['chunk_text'][:100]}...")
        print(f"\nAnswer: {answer}")

    return {"query": query, "retrieved_chunks": retrieved, "answer": answer}


if __name__ == "__main__":
    # Quick manual test
    result = answer_question("What is backpropagation used for?", verbose=True)