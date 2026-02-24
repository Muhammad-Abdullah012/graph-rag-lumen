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

ANSWER_SYSTEM = """You are a Eurocode structural engineering expert assistant.
You have been given CONTEXT retrieved directly from official Eurocode documents.

═══════════════════════════════════════════════════════════════
CRITICAL RULES — YOU MUST FOLLOW THESE WITHOUT EXCEPTION:
═══════════════════════════════════════════════════════════════

RULE 1 — USE ONLY THE CONTEXT:
  • Your answer MUST come EXCLUSIVELY from the provided CONTEXT.
  • DO NOT use any knowledge from your training data.
  • DO NOT invent, assume, or extrapolate ANY information.
  • If the exact answer is in the CONTEXT, use it. If not, say "Diese Information ist im bereitgestellten Kontext nicht vorhanden."

RULE 2 — COPY FORMULAS EXACTLY:
  • Every formula in the CONTEXT appears as LaTeX (e.g. R_{\\mathrm{d}} = ...).
  • Copy ALL relevant formulas VERBATIM from the CONTEXT — do NOT rewrite or simplify them.
  • Wrap every formula with $$ for display: $$R_{\\mathrm{d}} = \\frac{1}{\\gamma_{\\mathrm{Rd}}} R\\left\\{...\\right\\}$$
  • Wrap inline variables with $: the symbol $\\gamma_{\\mathrm{Rd}}$ represents...
  • NEVER write a formula that does not appear in the CONTEXT.

RULE 3 — INCLUDE ALL RELEVANT CONTENT:
  • Include ALL formulas from the CONTEXT that are relevant to the question.
  • Include table content if relevant.
  • Include figure descriptions if relevant.
  • Mention equation numbers like (6.6), (6.6a) etc. when present.

RULE 4 — LANGUAGE:
  • Match the user's language exactly (German question → German answer).
  • Technical terms from the CONTEXT should be quoted verbatim.

RULE 5 — CITATIONS:
  • Always cite: document name, section number (e.g. Abschnitt 6.3.5), page number.

RULE 6 — FORMAT:
  • Structure the answer clearly with the main formula first, then variable definitions.
  • Use numbered lists for multiple formulas or conditions.
  • For variable definitions, use bullet points: $\\gamma_{\\mathrm{Rd}}$ — Teilsicherheitsbeiwert für...
"""


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
    ref = f" (Dokument: {doc}, Seite: {page})" if doc else ""

    if category in ("sections", "semantic"):
        title = item.get("title", "Unknown")
        content = item.get("content", item.get("preview", ""))
        # Include formula LaTeX embedded in section (from enriched search)
        formula_latex = item.get("formula_latex", [])
        block = f"[{rank}] Abschnitt: {title}{ref}\n{content}"
        if formula_latex:
            formulas_str = "\n".join(
                f"  $$  {lat}  $$"
                for lat in formula_latex
                if lat and lat.strip()
            )
            if formulas_str:
                block += f"\n\nFormeln in diesem Abschnitt:\n{formulas_str}"
        return block

    if category == "formulas":
        latex = item.get("latex", "")
        section = item.get("section", "")
        return f"[{rank}] Formel in Abschnitt '{section}'{ref}\nLaTeX: $${latex}$$"

    if category == "tables":
        caption = item.get("caption", "")
        content = item.get("content", "")[:1200]
        return f"[{rank}] Tabelle: {caption}{ref}\n{content}"

    if category == "concepts":
        name = item.get("concept", "")
        desc = item.get("description", "")
        return f"[{rank}] Konzept: {name} — {desc}"

    if category == "figures":
        caption    = item.get("caption", "")
        desc       = item.get("description", "")
        annotation = item.get("annotation", "")
        image_path = item.get("image_path", "")
        body = desc
        if annotation and annotation.strip() and annotation.strip() != desc.strip():
            body = f"{desc}\nAnnotation: {annotation}".strip()
        block = f"[{rank}] Abbildung: {caption}{ref}\n{body}"
        if image_path:
            block += f"\n![{caption}]({image_path})"
        return block

    if category == "chapters":
        title = item.get("title", "")
        return f"[{rank}] Kapitel: {title}{ref}"

    # Fallback
    return f"[{rank}] {category}: {json.dumps(item, ensure_ascii=False, default=str)[:400]}"


def _rank_results(
    query: str,
    raw: Dict[str, List[Dict[str, Any]]],
    query_embedding: Optional[List[float]] = None,
    max_context_chars: int = 20000,
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
    q_lower = query.lower()
    is_formula = any(w in q_lower for w in (
        "formel", "formula", "gleichung", "berechnung", "berechnen",
        "equation", "calculate", r"\frac", r"\gamma", "latex",
        "ausgedrückt", "ausdruck", "berechnet", "ermittelt",
    ))
    is_figure = any(w in q_lower for w in (
        "abbildung", "bild", "figure", "diagram", "grafik",
        "diagramm", "querschnitt", "skizze", "chart",
    ))
    is_table = any(w in q_lower for w in (
        "tabelle", "table", "wert", "werte", "values", "parameter",
    ))

    # For formula-rich queries, heavily boost formulas and sections containing formulas
    CATEGORY_BONUS = {
        "sections":  2.5,
        "semantic":  2.0,
        "formulas":  3.0 if is_formula else 2.0,
        "tables":    2.5 if is_table   else 1.3,
        "figures":   2.5 if is_figure  else 0.8,
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

        # Extra boost for sections that have embedded formulas
        formula_boost = 0.5 if (cat in ("sections", "semantic") and item.get("formula_latex")) else 0.0

        bonus = CATEGORY_BONUS.get(cat, 0.5)
        combined = (bm * 1.0) + (neo_score * 1.5) + (emb_score * 2.0) + bonus + formula_boost
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

    return "\n\n".join(parts) if parts else "(Keine relevanten Ergebnisse im Wissensgraphen gefunden.)"


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
            max_context_chars=20000,
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
        context = state.get("ranked_context", "") or "(Keine relevanten Ergebnisse im Wissensgraphen gefunden.)"

        user_prompt = (
            f"KONTEXT AUS DEM WISSENSGRAPHEN:\n"
            f"{'=' * 60}\n"
            f"{context}\n"
            f"{'=' * 60}\n\n"
            f"WICHTIG: Deine Antwort MUSS ausschließlich auf dem obigen KONTEXT basieren.\n"
            f"Kopiere alle relevanten Formeln GENAU wie im KONTEXT angegeben (in $$...$$).\n"
            f"Erfinde KEINE Formeln, die nicht im KONTEXT stehen.\n\n"
            f"FRAGE: {question}"
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

    async def astream_answer(self, question: str):
        """Stream the agent's answer as an async generator of SSE-ready dicts.

        Yields dicts with one of three shapes:
          {"type": "status",  "step": str, "message": str}
          {"type": "token",   "content": str}
          {"type": "done",    "answer": str, "tools_used": list, "route": str}

        The non-LLM steps (routing, graph search, ranking) emit status events
        so the user sees progress instead of a blank loading screen.  The final
        LLM call streams tokens directly, giving word-by-word output.
        """
        # ── Step 1: classify / route ─────────────────────────────────
        yield {"type": "status", "step": "routing", "message": "Klassifiziere Anfrage…"}

        route = "eurocode"
        try:
            resp = await self.llm.ainvoke([
                SystemMessage(content=_ROUTER_SYSTEM),
                HumanMessage(content=question.strip()),
            ])
            label = (resp.content if hasattr(resp, "content") else str(resp)).strip().lower()
            if label.startswith("greeting"):
                route = "greeting"
        except Exception as e:
            logger.warning("Router call failed in stream, defaulting to eurocode: %s", e)

        # ── Greeting path ─────────────────────────────────────────────
        if route == "greeting":
            yield {"type": "status", "step": "greeting", "message": "Bereite Antwort vor…"}
            full_answer = ""
            try:
                async for chunk in self.llm.astream([
                    SystemMessage(content=GREETING_SYSTEM),
                    HumanMessage(content=question),
                ]):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        full_answer += token
                        yield {"type": "token", "content": token}
            except Exception as e:
                logger.error("Greeting stream failed: %s", e)
                fallback = "Hallo! Wie kann ich Ihnen helfen?"
                yield {"type": "token", "content": fallback}
                full_answer = fallback
            yield {"type": "done", "answer": full_answer, "tools_used": [], "route": "greeting"}
            return

        # ── Eurocode path ─────────────────────────────────────────────
        # Step 2: graph search
        yield {"type": "status", "step": "searching", "message": "Suche im Wissensgraphen…"}
        tools_used: List[Dict[str, Any]] = []
        raw: Dict[str, Any] = {}
        try:
            raw = self.querier.general_search(question)
            tools_used.append({"tool": "graph_search", "arguments": {"query": question}})
        except Exception as e:
            logger.error("Graph search failed in stream: %s", e)
            tools_used.append({"tool": "graph_search", "arguments": {"query": question}, "error": str(e)})

        # Step 3: rank context
        yield {"type": "status", "step": "ranking", "message": "Bewertet Suchergebnisse…"}
        query_embedding: Optional[List[float]] = None
        try:
            query_embedding = get_ollama_client().generate_embedding(question)
        except Exception:
            pass

        context = _rank_results(
            query=question,
            raw=raw,
            query_embedding=query_embedding,
            max_context_chars=20000,
        )
        tools_used.append({
            "tool": "rank_context",
            "arguments": {"top_items": context.count("["), "chars": len(context)},
        })

        # Step 4: stream LLM answer
        yield {"type": "status", "step": "answering", "message": "Generiere Antwort…"}

        user_prompt = (
            f"KONTEXT AUS DEM WISSENSGRAPHEN:\n"
            f"{'=' * 60}\n"
            f"{context}\n"
            f"{'=' * 60}\n\n"
            f"WICHTIG: Deine Antwort MUSS ausschließlich auf dem obigen KONTEXT basieren.\n"
            f"Kopiere alle relevanten Formeln GENAU wie im KONTEXT angegeben (in $$...$$).\n"
            f"Erfinde KEINE Formeln, die nicht im KONTEXT stehen.\n\n"
            f"FRAGE: {question}"
        )

        full_answer = ""
        try:
            async for chunk in self.llm.astream([
                SystemMessage(content=ANSWER_SYSTEM),
                HumanMessage(content=user_prompt),
            ]):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    full_answer += token
                    yield {"type": "token", "content": token}
        except Exception as e:
            logger.error("Answer LLM stream failed: %s", e)
            err = "Die Anfrage konnte aufgrund eines technischen Fehlers nicht verarbeitet werden."
            yield {"type": "token", "content": err}
            full_answer = err

        # Step 5: post-process assembled answer and emit done
        full_answer = _postprocess_latex(full_answer)
        yield {"type": "done", "answer": full_answer, "tools_used": tools_used, "route": "eurocode"}

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
