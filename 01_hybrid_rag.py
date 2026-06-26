"""
Hybrid RAG: Dense Vector Search + Sparse BM25 -> Reciprocal Rank Fusion -> LLM
Runs fully offline with fake embeddings and mock LLM responses.
"""""

import math
import re
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Sample corpus - a mini knowledge base about tech companies
# ---------------------------------------------------------------------------
DOCUMENTS = [
    {"id": 0, "text": "OpenAI was founded in 2015 by Sam Altman and Elon Musk. It develops large language models including GPT-4."},
    {"id": 1, "text": "Google DeepMind created AlphaGo, the first AI to defeat a world champion at the board game Go in 2016."},
    {"id": 2, "text": "Anthropic was founded in 2021 by former OpenAI researchers Dario Amodei and Daniela Amodei."},
    {"id": 3, "text": "Microsoft invested heavily in OpenAI and integrated GPT models into its Bing search engine and Office suite."},
    {"id": 4, "text": "Meta AI released the LLaMA family of open-weight language models, enabling on-device inference."},
    {"id": 5, "text": "Hugging Face hosts thousands of open-source machine learning models and datasets on its platform."},
    {"id": 6, "text": "NVIDIA GPUs power most modern AI training workloads; its H100 chip is widely used in data centers."},
    {"id": 7, "text": "Mistral AI is a Paris-based startup that released high-performance open-weight LLMs in 2023."},
]

QUERY = "Who founded OpenAI and what models do they make?"

# ---------------------------------------------------------------------------
# Fake embedding model (reproducible random vectors per word vocabulary)
# ---------------------------------------------------------------------------
class FakeEmbedder:
    """Deterministic fake embeddings using hashed word vectors."""

    DIM = 64

    def embed(self, text: str) -> np.ndarray:
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return np.zeros(self.DIM)
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        vec = rng.standard_normal(self.DIM)
        return vec / (np.linalg.norm(vec) + 1e-9)


# ---------------------------------------------------------------------------
# Dense vector store (in-memory cosine similarity)
# ---------------------------------------------------------------------------
class VectorStore:
    def __init__(self, embedder: FakeEmbedder):
        self.embedder = embedder
        self.index: list[tuple[int, np.ndarray]] = []

    def add_documents(self, docs: list[dict]):
        for doc in docs:
            vec = self.embedder.embed(doc["text"])
            self.index.append((doc["id"], vec))

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q_vec = self.embedder.embed(query)
        scores = []
        for doc_id, vec in self.index:
            score = float(np.dot(q_vec, vec))
            scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# BM25 sparse retriever (manual implementation, no external deps)
# ---------------------------------------------------------------------------
class BM25:
    K1 = 1.5
    B = 0.75

    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.N = len(docs)
        self.tokenized = [re.findall(r"\w+", d["text"].lower()) for d in docs]
        avg_len = sum(len(t) for t in self.tokenized) / max(self.N, 1)
        self.avgdl = avg_len
        self.df: dict[str, int] = defaultdict(int)
        for tokens in self.tokenized:
            for term in set(tokens):
                self.df[term] += 1

    def score(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q_terms = re.findall(r"\w+", query.lower())
        scores = []
        for idx, tokens in enumerate(self.tokenized):
            tf_map: dict[str, int] = defaultdict(int)
            for t in tokens:
                tf_map[t] += 1
            dl = len(tokens)
            s = 0.0
            for term in q_terms:
                tf = tf_map.get(term, 0)
                df = self.df.get(term, 0)
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
                numerator = tf * (self.K1 + 1)
                denominator = tf + self.K1 * (1 - self.B + self.B * dl / self.avgdl)
                s += idf * numerator / denominator
            scores.append((self.docs[idx]["id"], s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60
) -> list[tuple[int, float]]:
    fused: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------
def mock_llm(query: str, context_chunks: list[str]) -> str:
    context = "\n".join(f"- {c}" for c in context_chunks)
    return (
        f"[Mock LLM Answer]\n"
        f"Query: {query}\n\n"
        f"Based on the retrieved context:\n{context}\n\n"
        f"Answer: OpenAI was co-founded by Sam Altman and Elon Musk in 2015. "
        f"They are known for developing GPT-4 and other large language models."
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_hybrid_rag(query: str, top_k: int = 3):
    print("=" * 60)
    print("HYBRID RAG PIPELINE")
    print("=" * 60)
    print(f"Query: {query}\n")

    embedder = FakeEmbedder()
    vector_store = VectorStore(embedder)
    vector_store.add_documents(DOCUMENTS)
    bm25 = BM25(DOCUMENTS)

    # Step 1: Dense retrieval
    dense_results = vector_store.search(query, top_k=5)
    print("Step 1 - Dense (vector) retrieval:")
    for doc_id, score in dense_results:
        print(f"  doc_id={doc_id} score={score:.4f} text={DOCUMENTS[doc_id]['text'][:60]}...")

    # Step 2: Sparse retrieval
    sparse_results = bm25.score(query, top_k=5)
    print("\nStep 2 - Sparse (BM25) retrieval:")
    for doc_id, score in sparse_results:
        print(f"  doc_id={doc_id} score={score:.4f} text={DOCUMENTS[doc_id]['text'][:60]}...")

    # Step 3: Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion([dense_results, sparse_results])
    print("\nStep 3 - Reciprocal Rank Fusion (merged ranking):")
    for doc_id, score in fused[:top_k]:
        print(f"  doc_id={doc_id} rrf_score={score:.4f} text={DOCUMENTS[doc_id]['text'][:60]}...")

    # Step 4: Build context
    top_docs = [DOCUMENTS[doc_id]["text"] for doc_id, _ in fused[:top_k]]

    # Step 5: Generate answer
    answer = mock_llm(query, top_docs)
    print(f"\nStep 4 - LLM Answer:\n{answer}")
    print("=" * 60)


if __name__ == "__main__":
    run_hybrid_rag(QUERY)
