"""
Corrective RAG (CRAG): Query -> Retriever -> Evaluator/Grader -> Branch:
  CORRECT   -> LLM -> Answer
  AMBIGUOUS -> Query Rewriter -> re-retrieve -> LLM -> Answer
  INCORRECT -> Web Search Fallback -> LLM -> Answer
Runs fully offline with mock components.
"""

import re
from enum import Enum
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Sample corpus
# ---------------------------------------------------------------------------
DOCUMENTS = [
    {"id": 0, "text": "Anthropic was founded in 2021 by Dario Amodei and Daniela Amodei."},
    {"id": 1, "text": "Claude is Anthropic's AI assistant, designed with a focus on safety."},
    {"id": 2, "text": "The Amazon rainforest covers approximately 5.5 million square kilometres."},
    {"id": 3, "text": "Python is a high-level programming language known for its readability."},
    {"id": 4, "text": "GPT-4 was released by OpenAI in March 2023 and supports multimodal inputs."},
    {"id": 5, "text": "LangChain is a framework for building applications powered by language models."},
    {"id": 6, "text": "Retrieval-Augmented Generation (RAG) combines retrieval with generation for better accuracy."},
    {"id": 7, "text": "Vector databases like Pinecone and Weaviate store high-dimensional embeddings."},
]

WEB_RESULTS = {
    "who founded anthropic": "Anthropic was founded by Dario Amodei and Daniela Amodei in 2021 after they left OpenAI.",
    "anthropic funding": "Anthropic raised over $7 billion in funding from Google, Amazon, and others by 2024.",
    "claude capabilities": "Claude 3 supports 200k token context windows and excels at analysis and coding tasks.",
}

# ---------------------------------------------------------------------------
# Fake embedder
# ---------------------------------------------------------------------------
class FakeEmbedder:
    DIM = 48
    def embed(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        v = rng.standard_normal(self.DIM)
        return v / (np.linalg.norm(v) + 1e-9)

_embedder = FakeEmbedder()
_doc_vecs = [(d["id"], _embedder.embed(d["text"])) for d in DOCUMENTS]

# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = 3) -> list[dict]:
    q_vec = _embedder.embed(query)
    scores = [(doc_id, float(np.dot(q_vec, v))) for doc_id, v in _doc_vecs]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [DOCUMENTS[doc_id] for doc_id, _ in scores[:top_k]]

# ---------------------------------------------------------------------------
# Evaluator / Grader
# ---------------------------------------------------------------------------
class Verdict(Enum):
    CORRECT = "CORRECT"
    AMBIGUOUS = "AMBIGUOUS"
    INCORRECT = "INCORRECT"

def evaluate_relevance(query: str, docs: list[dict]) -> tuple[Verdict, float, str]:
    q_words = set(re.findall(r"\w+", query.lower()))
    doc_words = set()
    for d in docs:
        doc_words.update(re.findall(r"\w+", d["text"].lower()))
    overlap = len(q_words & doc_words) / max(len(q_words), 1)
    if overlap >= 0.45:
        return Verdict.CORRECT, overlap, f"High keyword overlap ({overlap:.0%})"
    elif overlap >= 0.20:
        return Verdict.AMBIGUOUS, overlap, f"Moderate overlap ({overlap:.0%}) - may need rewriting"
    else:
        return Verdict.INCORRECT, overlap, f"Low overlap ({overlap:.0%}) - retrieved docs are off-topic"

# ---------------------------------------------------------------------------
# Query rewriter
# ---------------------------------------------------------------------------
def rewrite_query(original_query: str) -> str:
    rewrites = {
        "who made anthropic": "who founded anthropic company",
        "tell me about claude": "claude AI assistant capabilities anthropic",
        "anthropic": "anthropic company founders history",
    }
    q_lower = original_query.lower()
    for key, rewrite in rewrites.items():
        if key in q_lower:
            return rewrite
    return original_query + " details overview background"

# ---------------------------------------------------------------------------
# Web search fallback
# ---------------------------------------------------------------------------
def web_search(query: str) -> list[str]:
    q_lower = query.lower()
    results = []
    for key, val in WEB_RESULTS.items():
        if any(w in q_lower for w in key.split()):
            results.append(val)
    return results if results else ["No relevant web results found for: " + query]

# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------
def mock_llm(query: str, context: list[str], source: str = "retrieval") -> str:
    ctx = "\n".join(f"  [{i+1}] {c}" for i, c in enumerate(context))
    return (
        f"[Mock LLM Answer - source: {source}]\n"
        f"Query: {query}\n\n"
        f"Context used:\n{ctx}\n\n"
        f"Answer: Based on the available context, Anthropic was founded in 2021 by "
        f"Dario Amodei and Daniela Amodei. Their flagship product, Claude, is an AI "
        f"assistant built with safety as a core design principle."
    )

# ---------------------------------------------------------------------------
# CRAG pipeline
# ---------------------------------------------------------------------------
def run_corrective_rag(query: str):
    print("=" * 60)
    print("CORRECTIVE RAG (CRAG) PIPELINE")
    print("=" * 60)
    print(f"Query: {query}\n")

    docs = retrieve(query)
    print("Step 1 - Initial Retrieval:")
    for d in docs:
        print(f"  [doc {d['id']}] {d['text'][:70]}...")

    verdict, confidence, reason = evaluate_relevance(query, docs)
    print(f"\nStep 2 - Evaluator Verdict: {verdict.value} ({reason})")

    if verdict == Verdict.CORRECT:
        print("\nStep 3 - Branch: CORRECT -> proceed to LLM")
        context = [d["text"] for d in docs]
        answer = mock_llm(query, context, source="initial retrieval")
    elif verdict == Verdict.AMBIGUOUS:
        print("\nStep 3 - Branch: AMBIGUOUS -> rewrite query and re-retrieve")
        rewritten = rewrite_query(query)
        print(f"  Rewritten query: '{rewritten}'")
        docs2 = retrieve(rewritten)
        print("  Re-retrieved docs:")
        for d in docs2:
            print(f"    [doc {d['id']}] {d['text'][:70]}...")
        context = [d["text"] for d in docs2]
        answer = mock_llm(query, context, source="re-retrieval after rewrite")
    else:
        print("\nStep 3 - Branch: INCORRECT -> web search fallback")
        web_hits = web_search(query)
        print("  Web search results:")
        for hit in web_hits:
            print(f"  * {hit}")
        answer = mock_llm(query, web_hits, source="web search fallback")

    print(f"\nStep 4 - Final Answer:\n{answer}")
    print("=" * 60)


if __name__ == "__main__":
    run_corrective_rag("Who founded Anthropic and what do they build?")
    print()
    run_corrective_rag("tell me about claude")
    print()
    run_corrective_rag("What is the capital of France and its population?")
