"""
Agentic RAG: Query -> Planner Agent -> Tool Selection Loop -> Reasoner -> Answer
Tools: VectorSearchTool, WebSearchTool, SQLDatabaseTool
Runs fully offline with mock tools and a rule-based planner.
"""

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Sample data stores
# ---------------------------------------------------------------------------
VECTOR_DOCS = [
    {"id": 0, "text": "OpenAI released GPT-4 in March 2023, showing significant capability improvements."},
    {"id": 1, "text": "Anthropic's Claude 3 Opus achieved top scores on several reasoning benchmarks."},
    {"id": 2, "text": "Google released Gemini Ultra in 2024, capable of multimodal reasoning."},
    {"id": 3, "text": "Meta's LLaMA 3 supports 128k context windows and runs on consumer hardware."},
    {"id": 4, "text": "Mistral 7B outperforms LLaMA 2 13B on most benchmarks despite being smaller."},
]

SQL_TABLE = [
    {"model": "GPT-4",        "company": "OpenAI",     "params_B": 1800, "release_year": 2023},
    {"model": "Claude 3 Opus", "company": "Anthropic",  "params_B": 200,  "release_year": 2024},
    {"model": "Gemini Ultra",  "company": "Google",     "params_B": 1000, "release_year": 2024},
    {"model": "LLaMA 3 70B",  "company": "Meta",       "params_B": 70,   "release_year": 2024},
    {"model": "Mistral 7B",   "company": "Mistral AI", "params_B": 7,    "release_year": 2023},
]

WEB_INDEX = {
    "latest AI models 2024": "In 2024, the leading AI models include GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, and LLaMA 3.",
    "benchmark comparison LLMs": "Claude 3 Opus leads on MMLU; GPT-4 leads on HumanEval; Gemini Ultra leads on multimodal tasks.",
    "open source LLMs": "LLaMA 3 and Mistral are the top open-source LLMs as of 2024.",
}

QUERY = "Which AI models were released in 2024, and how do they compare on benchmarks?"

# ---------------------------------------------------------------------------
# Fake embedder
# ---------------------------------------------------------------------------
class FakeEmbedder:
    DIM = 32
    def embed(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        v = rng.standard_normal(self.DIM)
        return v / (np.linalg.norm(v) + 1e-9)

_embedder = FakeEmbedder()
_doc_vecs = [(d["id"], _embedder.embed(d["text"])) for d in VECTOR_DOCS]

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@dataclass
class ToolResult:
    tool_name: str
    query_used: str
    results: Any
    confidence: float

class VectorSearchTool:
    name = "vector_search"
    description = "Semantic search over internal document corpus"
    def run(self, query: str, top_k: int = 3) -> ToolResult:
        q_vec = _embedder.embed(query)
        scores = [(doc_id, float(np.dot(q_vec, vec))) for doc_id, vec in _doc_vecs]
        scores.sort(key=lambda x: x[1], reverse=True)
        hits = [VECTOR_DOCS[doc_id]["text"] for doc_id, _ in scores[:top_k]]
        confidence = min(0.9, max(scores, key=lambda x: x[1])[1] + 0.5)
        return ToolResult(self.name, query, hits, confidence)

class WebSearchTool:
    name = "web_search"
    description = "Search the web for recent information"
    def run(self, query: str) -> ToolResult:
        best_key, best_score = None, 0
        q_words = set(query.lower().split())
        for key in WEB_INDEX:
            overlap = len(q_words & set(key.lower().split()))
            if overlap > best_score:
                best_score, best_key = overlap, key
        result = WEB_INDEX.get(best_key, "No relevant web results found.")
        confidence = 0.75 if best_score > 0 else 0.2
        return ToolResult(self.name, query, [result], confidence)

class SQLDatabaseTool:
    name = "sql_database"
    description = "Query structured database of AI model metadata"
    def run(self, query: str) -> ToolResult:
        rows = SQL_TABLE
        if "2024" in query:
            rows = [r for r in rows if r["release_year"] == 2024]
        if "open" in query.lower():
            rows = [r for r in rows if r["company"] in ("Meta", "Mistral AI")]
        formatted = [f"{r['model']} by {r['company']} ({r['params_B']}B params, {r['release_year']})" for r in rows]
        confidence = 0.85 if formatted else 0.1
        return ToolResult(self.name, query, formatted, confidence)

# ---------------------------------------------------------------------------
# Planner Agent
# ---------------------------------------------------------------------------
TOOLS = [VectorSearchTool(), WebSearchTool(), SQLDatabaseTool()]
TOOL_MAP = {t.name: t for t in TOOLS}

@dataclass
class PlanStep:
    tool_name: str
    sub_query: str

def planner_agent(query: str) -> list[PlanStep]:
    plan = []
    q = query.lower()
    plan.append(PlanStep("vector_search", query))
    if any(w in q for w in ["2024", "2023", "released", "params", "size"]):
        plan.append(PlanStep("sql_database", query))
    if any(w in q for w in ["benchmark", "compare", "latest", "recent", "best"]):
        plan.append(PlanStep("web_search", "benchmark comparison LLMs"))
    return plan

# ---------------------------------------------------------------------------
# Reasoner Agent
# ---------------------------------------------------------------------------
def reasoner_agent(query: str, tool_results: list[ToolResult]) -> str:
    context_parts = []
    for tr in tool_results:
        context_parts.append(f"[{tr.tool_name.upper()} - confidence={tr.confidence:.2f}]")
        for item in tr.results:
            context_parts.append(f"  * {item}")
    context = "\n".join(context_parts)
    return (
        f"[Mock Reasoner Agent Answer]\n"
        f"Query: {query}\n\n"
        f"Aggregated Tool Evidence:\n{context}\n\n"
        f"Final Answer: In 2024, notable AI models include Claude 3 Opus (Anthropic), "
        f"Gemini Ultra (Google), and LLaMA 3 (Meta). On benchmarks, Claude 3 Opus leads "
        f"on MMLU, GPT-4 leads on HumanEval, and Gemini Ultra excels at multimodal tasks."
    )

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def run_agentic_rag(query: str, max_iterations: int = 5):
    print("=" * 60)
    print("AGENTIC RAG PIPELINE")
    print("=" * 60)
    print(f"Query: {query}\n")

    plan = planner_agent(query)
    print(f"Step 1 - Planner selected {len(plan)} tool(s):")
    for step in plan:
        print(f"  -> {step.tool_name}: '{step.sub_query[:60]}'")

    all_results: list[ToolResult] = []
    for iteration, step in enumerate(plan[:max_iterations], start=1):
        tool = TOOL_MAP[step.tool_name]
        result = tool.run(step.sub_query)
        all_results.append(result)
        print(f"\nStep 2.{iteration} - {tool.name} results (confidence={result.confidence:.2f}):")
        for item in result.results:
            print(f"  * {item}")
        avg_confidence = sum(r.confidence for r in all_results) / len(all_results)
        if avg_confidence >= 0.80 and iteration >= 2:
            print(f"\n  [Agent] Sufficient confidence ({avg_confidence:.2f}). Stopping early.")
            break

    answer = reasoner_agent(query, all_results)
    print(f"\nStep 3 - Reasoner Agent:\n{answer}")
    print("=" * 60)


if __name__ == "__main__":
    run_agentic_rag(QUERY)
