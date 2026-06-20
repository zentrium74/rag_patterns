"""
GraphRAG: Query -> Entity Extraction -> Knowledge Graph -> Subgraph Retrieval
         -> Community Summaries -> LLM -> Answer
Runs fully offline using networkx and mock components.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

try:
    import networkx as nx
except ImportError:
    raise SystemExit("Install networkx: pip install networkx")

# ---------------------------------------------------------------------------
# Sample documents about tech companies
# ---------------------------------------------------------------------------
DOCUMENTS = [
    "Sam Altman is the CEO of OpenAI. OpenAI developed GPT-4, a large language model.",
    "Dario Amodei and Daniela Amodei founded Anthropic after leaving OpenAI in 2021.",
    "OpenAI is backed by Microsoft, which integrated GPT models into Azure and Bing.",
    "Anthropic developed Claude, an AI assistant focused on safety and helpfulness.",
    "Sam Altman previously served as president of Y Combinator, a startup accelerator.",
    "Google invested in Anthropic and also develops its own AI models through DeepMind.",
    "Mistral AI, headquartered in Paris, released open-weight LLMs and competes with OpenAI.",
    "NVIDIA supplies GPUs to both OpenAI and Anthropic for model training workloads.",
]

QUERY = "What is the relationship between Sam Altman and Anthropic?"

# ---------------------------------------------------------------------------
# Entity and relation types
# ---------------------------------------------------------------------------
@dataclass
class Entity:
    name: str
    entity_type: str  # Person / Company / Technology / Location / Project

@dataclass
class Relation:
    source: str
    relation: str  # works_at, founded, invested_in, developed, located_in, competes_with
    target: str
    context: str = ""

# ---------------------------------------------------------------------------
# Rule-based entity extractor (mock NER)
# ---------------------------------------------------------------------------
KNOWN_ENTITIES: dict[str, str] = {
    "Sam Altman": "Person",
    "Dario Amodei": "Person",
    "Daniela Amodei": "Person",
    "OpenAI": "Company",
    "Anthropic": "Company",
    "Microsoft": "Company",
    "Google": "Company",
    "NVIDIA": "Company",
    "DeepMind": "Company",
    "Mistral AI": "Company",
    "Y Combinator": "Company",
    "GPT-4": "Technology",
    "Claude": "Technology",
    "Azure": "Technology",
    "Bing": "Technology",
    "Paris": "Location",
}

RELATION_PATTERNS = [
    (r"(\w[\w ]+) is the CEO of ([\w ]+)", "ceo_of"),
    (r"(\w[\w ]+) founded ([\w ]+)", "founded"),
    (r"(\w[\w ]+) developed ([\w ]+)", "developed"),
    (r"(\w[\w ]+) invested in ([\w ]+)", "invested_in"),
    (r"(\w[\w ]+) is backed by ([\w ]+)", "backed_by"),
    (r"(\w[\w ]+) competes with ([\w ]+)", "competes_with"),
    (r"(\w[\w ]+) supplies .* to ([\w ]+)", "supplies_to"),
    (r"(\w[\w ]+) leaving ([\w ]+)", "left"),
    (r"([\w ]+) headquartered in ([\w ]+)", "located_in"),
    (r"(\w[\w ]+) integrated .* into ([\w ]+)", "integrated_into"),
]

def extract_entities(text: str) -> list[Entity]:
    found = []
    for name, etype in KNOWN_ENTITIES.items():
        if name.lower() in text.lower():
            found.append(Entity(name=name, entity_type=etype))
    return found

def extract_relations(text: str) -> list[Relation]:
    relations = []
    for pattern, rel_type in RELATION_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            src_raw, tgt_raw = m.group(1).strip(), m.group(2).strip()
            src = next((e for e in KNOWN_ENTITIES if e.lower() in src_raw.lower() or src_raw.lower() in e.lower()), src_raw)
            tgt = next((e for e in KNOWN_ENTITIES if e.lower() in tgt_raw.lower() or tgt_raw.lower() in e.lower()), tgt_raw)
            relations.append(Relation(source=src, relation=rel_type, target=tgt, context=text[:80]))
    return relations

# ---------------------------------------------------------------------------
# Knowledge graph builder
# ---------------------------------------------------------------------------
def build_knowledge_graph(docs: list[str]) -> nx.DiGraph:
    G = nx.DiGraph()
    for doc in docs:
        for ent in extract_entities(doc):
            G.add_node(ent.name, entity_type=ent.entity_type)
        for rel in extract_relations(doc):
            if rel.source not in G:
                G.add_node(rel.source, entity_type="Unknown")
            if rel.target not in G:
                G.add_node(rel.target, entity_type="Unknown")
            G.add_edge(rel.source, rel.target, relation=rel.relation, context=rel.context)
    return G

# ---------------------------------------------------------------------------
# Subgraph retrieval
# ---------------------------------------------------------------------------
def retrieve_subgraph(G: nx.DiGraph, query: str, hops: int = 2) -> nx.DiGraph:
    query_entities = [e for e in KNOWN_ENTITIES if e.lower() in query.lower()]
    seed_nodes = [n for n in G.nodes if any(e.lower() in n.lower() for e in query_entities)]
    visited = set(seed_nodes)
    frontier = set(seed_nodes)
    for _ in range(hops):
        new_frontier = set()
        for node in frontier:
            new_frontier.update(G.predecessors(node))
            new_frontier.update(G.successors(node))
        frontier = new_frontier - visited
        visited.update(frontier)
    return G.subgraph(visited).copy()

# ---------------------------------------------------------------------------
# Community summary
# ---------------------------------------------------------------------------
def summarize_subgraph(subG: nx.DiGraph) -> str:
    lines = []
    for u, v, data in subG.edges(data=True):
        u_type = subG.nodes[u].get("entity_type", "?")
        v_type = subG.nodes[v].get("entity_type", "?")
        lines.append(f"  [{u_type}] {u} --{data['relation']}--> [{v_type}] {v}")
    if not lines:
        return "No relevant subgraph found."
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------
def mock_llm(query: str, subgraph_summary: str) -> str:
    return (
        f"[Mock LLM Answer]\n"
        f"Query: {query}\n\n"
        f"Knowledge Graph Context:\n{subgraph_summary}\n\n"
        f"Answer: Sam Altman is the CEO of OpenAI. Dario Amodei and Daniela Amodei "
        f"left OpenAI to found Anthropic in 2021. So Sam Altman and Anthropic share "
        f"a historical connection through OpenAI, though he is not affiliated with Anthropic."
    )

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_graph_rag(query: str):
    print("=" * 60)
    print("GRAPH RAG PIPELINE")
    print("=" * 60)
    print(f"Query: {query}\n")

    G = build_knowledge_graph(DOCUMENTS)
    print(f"Step 1 - Knowledge Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("  Nodes:", list(G.nodes)[:10], "...")

    q_entities = extract_entities(query)
    print(f"\nStep 2 - Query entities detected: {[e.name for e in q_entities]}")

    subG = retrieve_subgraph(G, query, hops=2)
    print(f"\nStep 3 - Subgraph retrieved: {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges")

    summary = summarize_subgraph(subG)
    print(f"\nStep 4 - Subgraph / Community Summary:\n{summary}")

    answer = mock_llm(query, summary)
    print(f"\nStep 5 - LLM Answer:\n{answer}")
    print("=" * 60)


if __name__ == "__main__":
    run_graph_rag(QUERY)
