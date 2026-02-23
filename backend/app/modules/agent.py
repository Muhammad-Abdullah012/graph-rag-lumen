"""Eurocode Graph-RAG Agent — LangGraph state-machine implementation.

Flow
====
                     ┌─────────┐
            ┌────────│  Router  │────────┐
            │        └─────────┘        │
            ▼ (greeting/chitchat)       ▼ (eurocode question)
      ┌───────────┐              ┌─────────────┐
      │ Greet LLM │              │ Graph Search │
      └─────┬─────┘              └──────┬──────┘
            │                           │
            ▼                           ▼
          [END]                  ┌─────────────┐
                                │  Rank / Ctx  │
                                └──────┬──────┘
                                       │
                                       ▼
                                ┌─────────────┐
                                │  Answer LLM │
                                └──────┬──────┘
                                       │
                                       ▼
                                ┌─────────────┐
                                │  Post-proc  │
                                └──────┬──────┘
                                       │
                                       ▼
                                     [END]

Every eurocode question *always* hits the graph first (no reliance on the
LLM deciding to call a tool).  Results are ranked by relevance and the top
context is injected into the final LLM prompt.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from backend.app.modules.graph_querier import get_graph_querier, GraphQuerier
from backend.app.modules.ollama_client import get_ollama_client
from config.settings import settings

logger = logging.getLogger(__name__)


# ================================================================== #
#  State schema
# ================================================================== #

class AgentState(TypedDict, total=False):
    """Typed state flowing through the graph."""
    question: str                       # original user question
    route: str                          # "greeting" | "eurocode"
    search_results: Dict[str, Any]      # raw graph search results
    ranked_context: str                 # top context string for LLM
    tools_used: List[Dict[str, Any]]    # metadata about searches run
    answer: str                         # final answer text


# ================================================================== #
#  Prompts
# ================================================================== #

GREETING_SYSTEM = (
    "You are a friendly Eurocode structural-engineering assistant. "
    "The user sent a greeting, small-talk or a non-technical message. "
    "Reply briefly and warmly. Match the user's language (German → German, "
    "English → English). If the user writes in German, respond in German."
)

ANSWER_SYSTEM = (
    "You are a Eurocode structural engineering expert. "
    "Below is relevant context retrieved from a Graph-RAG knowledge graph "
    "containing Documents, Chapters, Sections, Tables, Figures, Formulas, "
    "and Concepts from Eurocode standards.\n\n"
    "RULES:\n"
    "1. Answer ONLY from the provided context. Do not invent information.\n"
    "2. Match the user's language (German → German, English → English).\n"
    "3. Cite the document name, section number, and page when possible.\n"
    "4. All mathematical formulas MUST use LaTeX dollar-sign delimiters:\n"
    "   - Inline: $...$\n"
    "   - Display / standalone: $$...$$\n"
    "   - NEVER use \\[...\\], \\(...\\), or bare LaTeX.\n"
    "   - Greek letters and variables must also be wrapped: "
    "$\\gamma_f$, $E_{\\mathrm{Ed}}$.\n"
    "5. Synthesize a clear, direct answer — do NOT dump raw search results.\n"
    "6. If the context does not contain the answer, say so honestly.\n"
)


# ================================================================== #
#  Result ranking helpers
# ================================================================== #

def _tokenize(text: str) -> List[str]:
    """Simple lowercased word tokenizer."""
    return re.findall(r"[a-zäöüß0-9_\\]+", text.lower())


def _bm25_score(
    query_tokens: List[str],
    doc_text: str,
    avg_dl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Simplified single-document BM25 score (no IDF — we don't have the
    full corpus stats, but query-term frequency in the doc is enough for
    ranking a small candidate set)."""
    doc_tokens = _tokenize(doc_text)
    dl = len(doc_tokens)
    if dl == 0:
        return 0.0
    tf = Counter(doc_tokens)
    score = 0.0
    for qt in query_tokens:
        f = tf.get(qt, 0)
        numerator = f * (k1 + 1)
        denominator = f + k1 * (1 - b + b * dl / max(avg_dl, 1))
        score += numerator / denominator if denominator else 0.0
    return score


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _item_text(item: Dict, category: str) -> str:
    """Extract the main searchable text from an item."""
    parts = []
    for key in ("title", "content", "preview", "section", "caption",
                "description", "concept", "latex", "formula"):
        v = item.get(key)
        if v and isinstance(v, str):
            parts.append(v)
    return " ".join(parts)


def _format_item(category: str, item: Dict, rank: int) -> str:
    """Format a single result item for the LLM context."""
    doc = item.get("document", "")
    page = item.get("page", "")
    ref = f" (Document: {doc}, Page: {page})" if doc else ""

    if category in ("sections", "semantic"):
        title = item.get("title", "Unknown")
        content = item.get("content", item.get("preview", ""))
        return f"[{rank}] Section: {title}{ref}\n{content}"

    if category == "formulas":
        latex = item.get("latex", "")
        section = item.get("section", "")
        return f"[{rank}] Formula in '{section}'{ref}\nLaTeX: $${latex}$$"

    if category == "tables":
        caption = item.get("caption", "")
        content = item.get("content", "")[:800]
        return f"[{rank}] Table: {caption}{ref}\n{content}"

    if category == "concepts":
        name = item.get("concept", "")
        desc = item.get("description", "")
        return f"[{rank}] Concept: {name} — {desc}"

    if category == "figures":
        caption    = item.get("caption", "")
        desc       = item.get("description", "")
        annotation = item.get("annotation", "")
        body = desc
        if annotation and annotation.strip() and annotation.strip() != desc.strip():
            body = f"{desc}\nAnnotation: {annotation}".strip()
        return f"[{rank}] Figure: {caption}{ref}\n{body}"

    if category == "chapters":
        title = item.get("title", "")
        return f"[{rank}] Chapter: {title}{ref}"

    # Fallback
    return f"[{rank}] {category}: {json.dumps(item, ensure_ascii=False, default=str)[:400]}"


def _rank_results(
    query: str,
    raw: Dict[str, List[Dict[str, Any]]],
    query_embedding: Optional[List[float]] = None,
    max_context_chars: int = 12000,
) -> str:
    """Rank all search results by relevance and build a context string.

    Scoring strategy (per item):
      • BM25 over the item's text   (lexical match)
      • Cosine similarity if embeddings are available   (semantic match)
      • Original score returned by Neo4j full-text index
      • Category bonus  (query-aware: formula/figure/table queries boost their
        respective categories so they are not crowded out by sections)

    Returns a formatted context string of the top items, truncated to
    *max_context_chars*.
    """
    # Detect query intent to rebalance category bonuses dynamically.
    # This prevents sections from monopolising the context window when the
    # user is specifically asking about a formula, figure, or table.
    q_lower = query.lower()
    is_formula = any(w in q_lower for w in (
        "formel", "formula", "gleichung", "berechnung", "berechnen",
        "equation", "calculate", r"\frac", r"\gamma", "latex",
    ))
    is_figure = any(w in q_lower for w in (
        "abbildung", "bild", "figure", "diagram", "grafik",
        "diagramm", "querschnitt", "skizze", "chart",
    ))
    is_table = any(w in q_lower for w in (
        "tabelle", "table", "wert", "werte", "values", "parameter",
    ))

    CATEGORY_BONUS = {
        "sections":  2.0,
        "semantic":  1.8,
        "formulas":  2.5 if is_formula else 1.5,
        "tables":    2.0 if is_table   else 1.3,
        "figures":   2.0 if is_figure  else 0.8,
        "concepts":  1.0,
        "chapters":  0.5,
    }

    query_tokens = _tokenize(query)

    # Flatten every result category into a unified scored list
    all_texts: List[str] = []
    items_with_text: List[tuple] = []

    for cat, items in raw.items():
        if not isinstance(items, list):
            continue
        for item in items:
            text = _item_text(item, cat)
            all_texts.append(text)
            items_with_text.append((cat, item, text))

    avg_dl = (sum(len(_tokenize(t)) for t in all_texts) / len(all_texts)) if all_texts else 100.0

    scored: List[tuple] = []  # (score, category, item)

    for cat, item, text in items_with_text:
        # BM25
        bm = _bm25_score(query_tokens, text, avg_dl)

        # Neo4j score (full-text or vector score already returned)
        neo_score = float(item.get("score", 0) or 0)

        # Embedding similarity (if available)
        emb_score = 0.0
        if query_embedding:
            item_emb = item.get("embedding")
            if item_emb and isinstance(item_emb, list):
                emb_score = _cosine_sim(query_embedding, item_emb)

        bonus = CATEGORY_BONUS.get(cat, 0.5)
        combined = (bm * 1.0) + (neo_score * 1.5) + (emb_score * 2.0) + bonus
        scored.append((combined, cat, item))

    # Sort descending by combined score
    scored.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate by title/section/caption
    seen_keys: set = set()
    unique: List[tuple] = []
    for s, cat, item in scored:
        key = (
            item.get("title")
            or item.get("section")
            or item.get("caption")
            or item.get("concept")
            or item.get("latex", "")[:80]
            or str(item)[:60]
        )
        if key not in seen_keys:
            seen_keys.add(key)
            unique.append((s, cat, item))

    # Build context string
    parts: List[str] = []
    total_len = 0
    for rank, (score, cat, item) in enumerate(unique, 1):
        block = _format_item(cat, item, rank)
        if total_len + len(block) > max_context_chars:
            break
        parts.append(block)
        total_len += len(block)

    return "\n\n".join(parts) if parts else "(No relevant results found in the knowledge graph.)"


# ================================================================== #
#  LaTeX post-processing
# ================================================================== #

_LATEX_MARKERS = re.compile(
    r"\\frac|\\mathrm|\\gamma|\\psi|\\alpha|\\beta|\\sigma|\\epsilon"
    r"|\\sum|\\int|\\sqrt|\\cdot|\\tag|\\left|\\right|\\text"
    r"|\^\{|_\{"
)


def _postprocess_latex(text: str) -> str:
    r"""Ensure any LaTeX that slipped through without delimiters is wrapped.

    • \[ ... \]  →  $$...$$
    • \( ... \)  →  $...$
    • Lines with LaTeX tokens but no $ → wrap in $$...$$
    """
    if not text:
        return text

    # Convert \[...\] → $$...$$
    text = re.sub(
        r"\\\[\s*(.+?)\s*\\\]",
        lambda m: f"$${m.group(1).strip()}$$",
        text,
        flags=re.S,
    )
    # Convert \(...\) → $...$
    text = re.sub(
        r"\\\(\s*(.+?)\s*\\\)",
        lambda m: f"${m.group(1).strip()}$",
        text,
        flags=re.S,
    )

    # Wrap lines that have LaTeX tokens but no dollar signs
    lines = text.split("\n")
    out: List[str] = []
    for ln in lines:
        stripped = ln.strip()
        if stripped and "$" not in ln and _LATEX_MARKERS.search(ln):
            out.append(f"$${stripped}$$")
        else:
            out.append(ln)

    return "\n".join(out)


# ================================================================== #
#  Router classifier (fast, no LLM — keyword-based)
# ================================================================== #

_ROUTER_SYSTEM = (
    "You are a message classifier for a structural-engineering assistant. "
    "Decide whether the user's message is a GREETING/CHITCHAT or a QUESTION "
    "that requires searching a technical knowledge base.\n\n"
    "GREETING/CHITCHAT includes: hellos, farewells, thanks, 'how are you', "
    "capability questions ('what can you do?'), and small-talk.\n\n"
    "QUESTION includes: any request for technical information, definitions, "
    "calculations, formulas, standards, or anything that might be answered "
    "by searching a knowledge base — even if vague or short.\n\n"
    "Respond with exactly one word: greeting  OR  question\n"
    "No explanation, no punctuation — just the single word."
)


def _classify_query(question: str, llm: Any) -> str:
    """Classify *question* using the LLM; returns 'greeting' or 'eurocode'.

    A trivial empty-string guard runs first.  The LLM is prompted to return
    exactly one word ('greeting' or 'question').  Any failure or ambiguous
    response defaults to 'eurocode' so the graph search always runs.
    """
    if not question.strip():
        return "greeting"

    try:
        resp = llm.invoke([
            SystemMessage(content=_ROUTER_SYSTEM),
            HumanMessage(content=question.strip()),
        ])
        label = (resp.content if hasattr(resp, "content") else str(resp)).strip().lower()
        if label.startswith("greeting"):
            logger.info("Router: '%s' → greeting", question[:60])
            return "greeting"
    except Exception as e:
        logger.warning("Router LLM call failed, defaulting to eurocode: %s", e)

    logger.info("Router: '%s' → eurocode", question[:60])
    return "eurocode"


# ================================================================== #
#  LangGraph node builders
# ================================================================== #

def _build_graph(querier: GraphQuerier, llm: ChatOllama) -> StateGraph:
    """Construct and compile the LangGraph state machine."""

    # ── Node: router ─────────────────────────────────────────────────
    def router(state: AgentState) -> AgentState:
        question = state["question"]
        route = _classify_query(question, llm)
        return {**state, "route": route}

    # ── Node: greeting ───────────────────────────────────────────────
    def greet(state: AgentState) -> AgentState:
        try:
            messages = [
                SystemMessage(content=GREETING_SYSTEM),
                HumanMessage(content=state["question"]),
            ]
            resp = llm.invoke(messages)
            answer = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            logger.error("Greeting LLM call failed: %s", e)
            answer = "Hallo! Wie kann ich Ihnen helfen?"
        return {**state, "answer": answer, "tools_used": []}

    # ── Node: graph search ───────────────────────────────────────────
    def graph_search(state: AgentState) -> AgentState:
        question = state["question"]
        tools_used: List[Dict[str, Any]] = []

        try:
            raw = querier.general_search(question)
            tools_used.append({"tool": "graph_search", "arguments": {"query": question}})
        except Exception as e:
            logger.error("Graph search failed for '%s': %s", question[:80], e)
            raw = {}
            tools_used.append({
                "tool": "graph_search",
                "arguments": {"query": question},
                "error": str(e),
            })

        return {
            **state,
            "search_results": raw,
            "tools_used": tools_used,
        }

    # ── Node: rank & build context ───────────────────────────────────
    def rank_context(state: AgentState) -> AgentState:
        question = state["question"]
        raw = state.get("search_results", {})

        # Try to get query embedding for semantic ranking
        query_embedding: Optional[List[float]] = None
        try:
            query_embedding = get_ollama_client().generate_embedding(question)
        except Exception as e:
            logger.debug("Embedding for ranking unavailable: %s", e)

        context = _rank_results(
            query=question,
            raw=raw,
            query_embedding=query_embedding,
            max_context_chars=12000,
        )

        tools_used = list(state.get("tools_used", []))
        tools_used.append({
            "tool": "rank_context",
            "arguments": {"top_items": context.count("["), "chars": len(context)},
        })

        return {**state, "ranked_context": context, "tools_used": tools_used}

    # ── Node: answer LLM ────────────────────────────────────────────
    def answer_llm(state: AgentState) -> AgentState:
        question = state["question"]
        context = state.get("ranked_context", "") or "(No relevant results found in the knowledge graph.)"

        user_prompt = (
            f"CONTEXT FROM KNOWLEDGE GRAPH:\n"
            f"{'=' * 60}\n"
            f"{context}\n"
            f"{'=' * 60}\n\n"
            f"USER QUESTION: {question}"
        )

        try:
            messages = [
                SystemMessage(content=ANSWER_SYSTEM),
                HumanMessage(content=user_prompt),
            ]
            resp = llm.invoke(messages)
            answer = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            logger.error("Answer LLM call failed: %s", e)
            answer = (
                "Die Anfrage konnte aufgrund eines technischen Fehlers nicht verarbeitet werden. "
                "Bitte versuchen Sie es erneut."
            )
        return {**state, "answer": answer}

    # ── Node: post-process ──────────────────────────────────────────
    def postprocess(state: AgentState) -> AgentState:
        answer = _postprocess_latex(state.get("answer", ""))
        return {**state, "answer": answer}

    # ── Conditional edge from router ────────────────────────────────
    def route_decision(state: AgentState) -> str:
        return state.get("route", "eurocode")

    # ── Build the graph ─────────────────────────────────────────────
    graph = StateGraph(AgentState)

    graph.add_node("router", router)
    graph.add_node("greet", greet)
    graph.add_node("graph_search", graph_search)
    graph.add_node("rank_context", rank_context)
    graph.add_node("answer_llm", answer_llm)
    graph.add_node("postprocess", postprocess)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "greeting": "greet",
            "eurocode": "graph_search",
        },
    )

    graph.add_edge("greet", END)
    graph.add_edge("graph_search", "rank_context")
    graph.add_edge("rank_context", "answer_llm")
    graph.add_edge("answer_llm", "postprocess")
    graph.add_edge("postprocess", END)

    return graph.compile()


# ================================================================== #
#  Agent class
# ================================================================== #

class EurocodeAgent:
    """LangGraph-based Eurocode Graph-RAG agent."""

    def __init__(self):
        self.querier = get_graph_querier()
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_llm_model,
            temperature=0,
        )
        self.compiled = _build_graph(self.querier, self.llm)
        logger.info(
            "EurocodeAgent (LangGraph) initialized, model=%s",
            settings.ollama_llm_model,
        )

    async def aanswer(self, question: str) -> Dict[str, Any]:
        """Run the graph and return the final answer + metadata."""
        initial_state: AgentState = {
            "question": question,
            "route": "",
            "search_results": {},
            "ranked_context": "",
            "tools_used": [],
            "answer": "",
        }

        result = await self.compiled.ainvoke(initial_state)

        return {
            "answer": result.get("answer", ""),
            "tools_used": result.get("tools_used", []),
            "route": result.get("route", ""),
        }

    def answer_sync(self, question: str) -> Dict[str, Any]:
        """Synchronous variant."""
        initial_state: AgentState = {
            "question": question,
            "route": "",
            "search_results": {},
            "ranked_context": "",
            "tools_used": [],
            "answer": "",
        }

        result = self.compiled.invoke(initial_state)

        return {
            "answer": result.get("answer", ""),
            "tools_used": result.get("tools_used", []),
            "route": result.get("route", ""),
        }


# ================================================================== #
#  Singleton
# ================================================================== #

_agent: Optional[EurocodeAgent] = None


def get_agent() -> EurocodeAgent:
    global _agent
    if _agent is None:
        _agent = EurocodeAgent()
    return _agent
