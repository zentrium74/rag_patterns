"""
Multimodal RAG: Text Chunks + Images/Charts + Tables -> Shared Embedding Space
              -> Unified Vector Index -> Retrieval -> Multimodal LLM -> Answer
Runs fully offline with mock CLIP-style embeddings and a mock multimodal LLM.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Modality types
# ---------------------------------------------------------------------------
class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"

# ---------------------------------------------------------------------------
# Sample multimodal corpus
# ---------------------------------------------------------------------------
@dataclass
class Document:
    doc_id: str
    modality: Modality
    content: Any
    description: str

TEXT_CORPUS = [
    Document("t1", Modality.TEXT,
             "OpenAI GPT-4 was released in March 2023 with multimodal capabilities including image understanding.",
             "Text: GPT-4 release description"),
    Document("t2", Modality.TEXT,
             "Anthropic's Claude 3 family includes Haiku, Sonnet, and Opus, targeting different speed/quality tradeoffs.",
             "Text: Claude 3 model family description"),
    Document("t3", Modality.TEXT,
             "Google Gemini Ultra achieves state-of-the-art results on MMLU benchmark, surpassing human experts.",
             "Text: Gemini Ultra benchmark description"),
    Document("t4", Modality.TEXT,
             "Meta LLaMA 3 70B is an open-weight model competitive with proprietary models on reasoning tasks.",
             "Text: LLaMA 3 description"),
]

IMAGE_CORPUS = [
    Document("i1", Modality.IMAGE,
             {"filename": "gpt4_architecture_diagram.png",
              "caption": "GPT-4 transformer architecture with multimodal input encoder",
              "tags": ["architecture", "transformer", "GPT-4", "diagram"]},
             "Image: GPT-4 architecture diagram"),
    Document("i2", Modality.IMAGE,
             {"filename": "benchmark_comparison_chart.png",
              "caption": "Bar chart comparing GPT-4, Claude 3, Gemini on MMLU, HumanEval, and GSM8K benchmarks",
              "tags": ["benchmark", "comparison", "chart", "GPT-4", "Claude", "Gemini"]},
             "Image: Benchmark comparison bar chart"),
    Document("i3", Modality.IMAGE,
             {"filename": "llm_scaling_laws.png",
              "caption": "Log-log plot showing LLM performance scaling with compute and parameter count",
              "tags": ["scaling", "compute", "parameters", "research"]},
             "Image: LLM scaling laws plot"),
]

TABLE_CORPUS = [
    Document("tb1", Modality.TABLE,
             {"title": "LLM Model Comparison",
              "headers": ["Model", "Company", "Params (B)", "Context (K)", "Year"],
              "rows": [
                  ["GPT-4", "OpenAI", "~1800", "128", "2023"],
                  ["Claude 3 Opus", "Anthropic", "~200", "200", "2024"],
                  ["Gemini Ultra", "Google", "~1000", "1000", "2024"],
                  ["LLaMA 3 70B", "Meta", "70", "8", "2024"],
                  ["Mistral 7B", "Mistral", "7", "32", "2023"],
              ]},
             "Table: LLM model comparison with parameters and context length"),
    Document("tb2", Modality.TABLE,
             {"title": "Benchmark Scores",
              "headers": ["Model", "MMLU (%)", "HumanEval (%)", "GSM8K (%)"],
              "rows": [
                  ["GPT-4", "86.4", "67.0", "92.0"],
                  ["Claude 3 Opus", "86.8", "84.9", "95.0"],
                  ["Gemini Ultra", "90.0", "74.4", "94.4"],
                  ["LLaMA 3 70B", "82.0", "81.7", "93.0"],
              ]},
             "Table: Benchmark scores across MMLU, HumanEval, GSM8K"),
]

ALL_DOCUMENTS = TEXT_CORPUS + IMAGE_CORPUS + TABLE_CORPUS
QUERY = "How do GPT-4 and Claude 3 compare on benchmarks? Show me charts or tables."

# ---------------------------------------------------------------------------
# Mock CLIP-style multimodal embedder
# ---------------------------------------------------------------------------
class MultimodalEmbedder:
    DIM = 64
    MODALITY_OFFSET = {
        Modality.TEXT:  np.zeros(64),
        Modality.IMAGE: np.ones(64) * 0.05,
        Modality.TABLE: np.ones(64) * 0.03,
    }

    def _text_for_embedding(self, doc: Document) -> str:
        if doc.modality == Modality.TEXT:
            return doc.content
        elif doc.modality == Modality.IMAGE:
            tags = " ".join(doc.content.get("tags", []))
            return doc.content.get("caption", "") + " " + tags
        else:
            header_str = " ".join(doc.content.get("headers", []))
            row_str = " ".join(" ".join(r) for r in doc.content.get("rows", []))
            return doc.content.get("title", "") + " " + header_str + " " + row_str

    def embed(self, doc: Document) -> np.ndarray:
        text = self._text_for_embedding(doc)
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        vec = rng.standard_normal(self.DIM)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        offset = self.MODALITY_OFFSET[doc.modality]
        vec = vec + offset
        return vec / (np.linalg.norm(vec) + 1e-9)

    def embed_query(self, query: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(query)) % (2**31))
        vec = rng.standard_normal(self.DIM)
        return vec / (np.linalg.norm(vec) + 1e-9)

# ---------------------------------------------------------------------------
# Unified multimodal vector index
# ---------------------------------------------------------------------------
class MultimodalVectorIndex:
    def __init__(self, embedder: MultimodalEmbedder):
        self.embedder = embedder
        self.index: list[tuple[Document, np.ndarray]] = []

    def add_documents(self, docs: list[Document]):
        for doc in docs:
            vec = self.embedder.embed(doc)
            self.index.append((doc, vec))

    def search(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        q_vec = self.embedder.embed_query(query)
        scores = []
        for doc, vec in self.index:
            score = float(np.dot(q_vec, vec))
            scores.append((doc, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# ---------------------------------------------------------------------------
# Multimodal context formatter
# ---------------------------------------------------------------------------
def format_document(doc: Document) -> str:
    if doc.modality == Modality.TEXT:
        return f"[TEXT] {doc.content}"
    elif doc.modality == Modality.IMAGE:
        c = doc.content
        return f"[IMAGE] {c['filename']} - {c['caption']}"
    else:
        c = doc.content
        header = " | ".join(c["headers"])
        rows = "\n".join("  " + " | ".join(r) for r in c["rows"])
        return f"[TABLE] {c['title']}\n  {header}\n{rows}"

# ---------------------------------------------------------------------------
# Mock Multimodal LLM
# ---------------------------------------------------------------------------
def mock_multimodal_llm(query: str, retrieved: list[tuple[Document, float]]) -> str:
    modality_counts = {m: 0 for m in Modality}
    for doc, _ in retrieved:
        modality_counts[doc.modality] += 1
    context_lines = []
    for doc, score in retrieved:
        context_lines.append(f"  (score={score:.3f}) {format_document(doc)}")
    context = "\n".join(context_lines)
    return (
        f"[Mock Multimodal LLM Answer]\n"
        f"Query: {query}\n\n"
        f"Retrieved Context ({len(retrieved)} items - "
        f"text={modality_counts[Modality.TEXT]}, "
        f"images={modality_counts[Modality.IMAGE]}, "
        f"tables={modality_counts[Modality.TABLE]}):\n"
        f"{context}\n\n"
        f"Answer: Based on the benchmark comparison table and chart, Claude 3 Opus leads "
        f"on MMLU (86.8%) and HumanEval (84.9%), while GPT-4 scores 86.4% on MMLU and "
        f"67.0% on HumanEval. The bar chart visually confirms Claude 3 Opus outperforms "
        f"GPT-4 on coding benchmarks. Gemini Ultra achieves the highest MMLU score (90.0%)."
    )

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_multimodal_rag(query: str, top_k: int = 5):
    print("=" * 60)
    print("MULTIMODAL RAG PIPELINE")
    print("=" * 60)
    print(f"Query: {query}\n")

    embedder = MultimodalEmbedder()
    index = MultimodalVectorIndex(embedder)
    index.add_documents(ALL_DOCUMENTS)
    print(f"Step 1 - Unified Index built:")
    print(f"  {len(TEXT_CORPUS)} text chunks | {len(IMAGE_CORPUS)} images | {len(TABLE_CORPUS)} tables")
    print(f"  Total: {len(ALL_DOCUMENTS)} documents indexed in shared {embedder.DIM}-dim space")

    q_lower = query.lower()
    wants_image = any(w in q_lower for w in ["chart", "diagram", "plot", "image", "show", "visual"])
    wants_table = any(w in q_lower for w in ["table", "comparison", "compare", "number", "score"])
    print(f"\nStep 2 - Query Modality Preferences:")
    print(f"  Wants images/charts: {wants_image}")
    print(f"  Wants tables: {wants_table}")

    all_results = index.search(query, top_k=top_k)
    print(f"\nStep 3 - Unified Retrieval (top {top_k} across all modalities):")
    for doc, score in all_results:
        print(f"  [{doc.modality.value.upper():5s}] score={score:.3f} {doc.description}")

    boosted: list[tuple[Document, float]] = []
    for doc, score in all_results:
        boost = 1.0
        if wants_image and doc.modality == Modality.IMAGE:
            boost = 1.15
        if wants_table and doc.modality == Modality.TABLE:
            boost = 1.10
        boosted.append((doc, score * boost))
    boosted.sort(key=lambda x: x[1], reverse=True)
    final_results = boosted[:top_k]
    print(f"\nStep 4 - After Modality Boosting (top {top_k}):")
    for doc, score in final_results:
        print(f"  [{doc.modality.value.upper():5s}] score={score:.3f} {doc.description}")

    answer = mock_multimodal_llm(query, final_results)
    print(f"\nStep 5 - Multimodal LLM Answer:\n{answer}")
    print("=" * 60)


if __name__ == "__main__":
    run_multimodal_rag(QUERY)
