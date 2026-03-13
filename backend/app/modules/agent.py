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
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from backend.app.modules.graph_querier import get_graph_querier, GraphQuerier
from config.settings import settings

logger = logging.getLogger(__name__)

_DEBUG_LOG = os.path.join(os.path.dirname(__file__), "..", "..", "..", "debug_raw.jsonl")


def _strip_embeddings(obj: Any) -> Any:
    """Recursively remove 'embedding' keys (768-dim float lists) from search results."""
    if isinstance(obj, dict):
        return {k: _strip_embeddings(v) for k, v in obj.items() if k != "embedding"}
    if isinstance(obj, list):
        return [_strip_embeddings(v) for v in obj]
    return obj


def _write_debug(entry: Dict[str, Any]) -> None:
    """Append one JSON line to the debug log file (best-effort, never raises)."""
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.debug("Debug log write failed: %s", exc)


# ================================================================== #
#  State schema
# ================================================================== #

class AgentState(TypedDict, total=False):
    """Typed state flowing through the graph."""
    question: str                       # original user question (any language)
    search_query: str                   # German translation used for graph search
    keywords: str                       # LLM-extracted technical keywords for BM25
    route: str                          # "greeting" | "eurocode"
    search_results: Dict[str, Any]      # raw graph search results
    ranked_context: str                 # top context string for LLM
    img_mapping: Dict[str, str]         # IMG-N → real ![caption](url) mapping (OCR-embedded)
    extra_figures: List[str]            # Additional figures to append after answer (markdown lines)
    _extra_figures_raw: List[Dict[str, Any]]  # Raw figure dicts from graph enrichment
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

ANSWER_SYSTEM = """You are a Eurocode structural engineering assistant.
You are given CONTEXT pages retrieved from official Eurocode documents.
Answer the user's QUESTION using ONLY what is in the CONTEXT.

STEP 1 — EXTRACT (do this silently before writing your answer):
Scan every CONTEXT page and find all sentences, values, or formulas
that are relevant to the QUESTION. If the question asks for a value
or factor, look for it under ALL possible names.

STEP 2 — ANSWER:
Write your answer based ONLY on what you found in STEP 1.
- If STEP 1 found values/formulas → report them directly. Do NOT
  describe how they are calculated. Just state what the context says.
- If STEP 1 found nothing → use the fallback below.

READING ORDER:
- Read ALL context pages before forming your answer.
- Every page has relevance score in brackets (e.g. [Relevanz: 95.3]) — higher means more relevant.
- Supporting details, definitions, and cross-references may appear on later pages.

FORMULAS:
- Copy $...$ and $$...$$ blocks EXACTLY as they appear in the CONTEXT.
- Do NOT write any formula or symbol not literally present in the CONTEXT.
- Do NOT rewrite, simplify, rearrange, or paraphrase any formula.

TABLES:
- Copy pipe-delimited markdown tables EXACTLY.
- Always keep the header row and separator row (| --- | --- |).

NOTATION:
- Questions may use shorthand (e.g. "λ-Werte") that appears in the
  context as symbolic notation (e.g. $\\lambda_{\\mathrm{v},1}$).
- Treat these as the same topic. Do NOT require an exact string match.

LANGUAGE:
- Reply in the same language as the QUESTION.

IF NOT IN CONTEXT:
- Only use this fallback if STEP 1 found ZERO relevant content.
  German: "Diese Information ist im bereitgestellten Kontext nicht vorhanden."
  English: "This information is not available in the provided context."
- Do NOT use this fallback if ANY value, formula, or condition was found.
- Do NOT add general knowledge. Do NOT describe calculation procedures
  unless the context itself describes them.
"""


# ================================================================== #
#  Context formatting
# ================================================================== #


_IMG_RE = re.compile(r'!\[([^\]]*)\]\((/api/images/[^)]+)\)')


def _replace_images_with_placeholders(
    text: str, mapping: Dict[str, str], counter: List[int]
) -> str:
    """Replace ``![caption](/api/images/...)`` with ``[IMG-N]`` placeholders.

    *mapping* is mutated in-place: ``{"IMG-1": "![caption](/api/images/...)", ...}``.
    *counter* is a one-element list ``[next_number]`` so it survives across calls.
    """
    def _replacer(m: re.Match) -> str:
        key = f"IMG-{counter[0]}"
        counter[0] += 1
        mapping[key] = m.group(0)  # full ``![caption](url)``
        return f"[{key}]"
    return _IMG_RE.sub(_replacer, text)


def _restore_image_placeholders(answer: str, mapping: Dict[str, str]) -> str:
    """Replace ``[IMG-N]`` tokens in the LLM answer with real image markdown."""
    for key, md in mapping.items():
        answer = answer.replace(f"[{key}]", md)
    return answer


def _format_pages_as_context(
    pages: List[Dict[str, Any]],
    img_mapping: Optional[Dict[str, str]] = None,
    img_counter: Optional[List[int]] = None,
) -> str:
    """Format a list of pages as context for the LLM.

    Each page entry already contains the full OCR text of that page —
    formulas, tables, and figures are all inline in the page content.

    When *img_mapping* / *img_counter* are provided, inline image
    references are replaced with short ``[IMG-N]`` placeholders to
    prevent the LLM from hallucinating corrupted URLs.
    """
    if not pages:
        return "(Keine relevanten Seiten im Wissensgraphen gefunden.)"

    # Trim pages to fit within the context budget (lower-ranked pages first).
    budget = settings.total_context_budget
    if budget > 0:
        trimmed: List[Dict[str, Any]] = []
        used = 0
        for page in pages:
            page_len = len(page.get("content", ""))
            if used + page_len > budget and trimmed:
                break
            trimmed.append(page)
            used += page_len
        pages = trimmed

    parts: List[str] = []
    for i, page in enumerate(pages, 1):
        page_num = page.get("page_number", "?")
        chapter  = page.get("chapter", "")
        doc      = page.get("document", "")
        content  = page.get("content", "")

        if img_mapping is not None and img_counter is not None:
            content = _replace_images_with_placeholders(
                content, img_mapping, img_counter
            )

        ref      = f" (Dokument: {doc})" if doc else ""
        chap_str = f" | {chapter}" if chapter else ""
        header   = f"[{i}] Seite {page_num}{chap_str}{ref}"
        if "rerank_score" in page:
            header += f" [Relevanz: {page['rerank_score']:.1f}]"
        parts.append(f"{header}\n{content}")

    return "\n\n".join(parts)


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

    # Don't touch anything already wrapped in $$...$$
    # Split on existing $$-blocks, only process the non-math parts
    parts = re.split(r'(\$\$[\s\S]*?\$\$)', text)
    
    processed = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # This is already a $$...$$ block — leave it alone
            processed.append(part)
            continue

        # Convert \[...\] → $$...$$
        part = re.sub(
            r"\\\[\s*(.+?)\s*\\\]",
            lambda m: f"\n$${m.group(1).strip()}$$\n",
            part,
            flags=re.S,
        )
        # Convert \(...\) → $$...$$
        part = re.sub(
            r"\\\(\s*(.+?)\s*\\\)",
            lambda m: f"\n$${m.group(1).strip()}$$\n",
            part,
            flags=re.S,
        )

        # Wrap lines that have LaTeX tokens but no dollar signs
        lines = part.split("\n")
        out: List[str] = []
        for ln in lines:
            stripped = ln.strip()
            if stripped and "$" not in ln and _LATEX_MARKERS.search(ln):
                out.append(f"$${stripped}$$")
            else:
                out.append(ln)
        processed.append("\n".join(out))

    return "".join(processed)

# ================================================================== #
#  Router classifier (fast, no LLM — keyword-based)
# ================================================================== #

_ROUTER_SYSTEM = (
    "You are a message classifier for a structural-engineering assistant. "
    "The knowledge base contains ONLY Eurocode standards for structural "
    "engineering (steel, concrete, bridges, loads, design, fatigue, etc.).\n\n"
    "Classify the message into exactly one category:\n\n"
    "greeting — hellos, farewells, thanks, 'how are you', capability questions, small-talk.\n\n"
    "question — requests about structural engineering, Eurocodes, civil engineering, "
    "building codes, loads, materials, steel, concrete, timber, bridges, foundations, "
    "seismic design, fatigue, formulas, safety factors, or any topic that could "
    "plausibly appear in a Eurocode standard.\n\n"
    "offtopic — anything clearly unrelated to structural engineering or Eurocodes: "
    "astronomy, cooking, history, geography, politics, biology, general science "
    "questions, sports, entertainment, etc.\n\n"
    "Respond with exactly one word: greeting  OR  question  OR  offtopic\n"
    "No explanation, no punctuation — just the single word."
)


_TRANSLATE_SYSTEM = (
    "You are a technical translator specializing in structural engineering and Eurocodes. "
    "Translate the following query into German using standard Eurocode terminology. "
    "If the query is already in German, return it exactly as-is. "
    "Return ONLY the German text — no explanation, no quotes, no labels.\n\n"
    "Key Eurocode term mappings:\n"
    # General structural terms
    "  design value → Bemessungswert\n"
    "  characteristic value → charakteristischer Wert\n"
    "  partial factor / partial safety factor → Teilsicherheitsbeiwert\n"
    "  load combination → Lastkombination\n"
    "  combination value → Kombinationswert\n"
    "  frequent value → häufiger Wert\n"
    "  quasi-permanent value → quasi-ständiger Wert\n"
    "  ultimate limit state (ULS) → Grenzzustand der Tragfähigkeit\n"
    "  serviceability limit state (SLS) → Grenzzustand der Gebrauchstauglichkeit\n"
    "  National Annex → Nationaler Anhang\n"
    "  nationally determined parameter (NDP) → national bestimmter Parameter\n"
    # Loads
    "  dead load / self-weight → Eigengewicht\n"
    "  live load / imposed load → Nutzlast\n"
    "  wind load → Windlast\n"
    "  snow load → Schneelast\n"
    "  traffic load → Verkehrslast\n"
    "  temperature action → Temperatureinwirkung\n"
    "  accidental action → außergewöhnliche Einwirkung\n"
    "  seismic action → Erdbebeneinwirkung\n"
    # Materials
    "  steel → Stahl\n"
    "  concrete → Beton\n"
    "  reinforcement → Bewehrung\n"
    "  timber → Holz\n"
    "  masonry → Mauerwerk\n"
    "  elastic modulus / modulus of elasticity → Elastizitätsmodul\n"
    "  shear modulus → Schubmodul\n"
    "  yield strength → Streckgrenze\n"
    "  tensile strength → Zugfestigkeit\n"
    "  Poisson's ratio → Querdehnzahl\n"
    "  coefficient of thermal expansion → Wärmeausdehnungskoeffizient\n"
    "  density → Wichte / Rohdichte\n"
    "  material constant → Materialkonstante\n"
    # Steel-specific
    "  buckling → Knicken\n"
    "  lateral torsional buckling → Biegedrillknicken\n"
    "  fatigue → Ermüdung\n"
    "  notch case / notch category → Kerbfall\n"
    "  weld / welding → Schweißnaht\n"
    "  cross-section class → Querschnittsklasse\n"
    "  dynamic amplification factor → Schwingbeiwert\n"
    # Concrete-specific
    "  crack width → Rissbreite\n"
    "  creep → Kriechen\n"
    "  shrinkage → Schwinden\n"
    "  prestress / prestressed → Vorspannung\n"
    "  anchorage → Verankerung\n"
    # Structural elements
    "  bridge → Brücke\n"
    "  road bridge → Straßenbrücke\n"
    "  railway bridge → Eisenbahnbrücke\n"
    "  column → Stütze\n"
    "  beam → Träger\n"
    "  section → Abschnitt"
)

def _translate_to_german(question: str, llm: Any) -> str:
    """Translate *question* to German for graph search.  If already German, returns as-is."""
    try:
        resp = llm.invoke([
            SystemMessage(content=_TRANSLATE_SYSTEM),
            HumanMessage(content=question.strip()),
        ])
        translated = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        if translated:
            logger.info("Search query translated: '%s' → '%s'", question[:60], translated[:60])
            return translated
    except Exception as e:
        logger.warning("Translation failed, using original query: %s", e)
    return question


async def _atranslate_to_german(question: str, llm: Any) -> str:
    """Async version of _translate_to_german."""
    try:
        resp = await llm.ainvoke([
            SystemMessage(content=_TRANSLATE_SYSTEM),
            HumanMessage(content=question.strip()),
        ])
        translated = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        if translated:
            logger.info("Search query translated: '%s' → '%s'", question[:60], translated[:60])
            return translated
    except Exception as e:
        logger.warning("Translation failed, using original query: %s", e)
    return question


def _classify_query(question: str, llm: Any) -> str:
    """Classify *question* using the LLM; returns 'greeting', 'offtopic', or 'eurocode'.

    A trivial empty-string guard runs first.  The LLM is prompted to return
    exactly one word ('greeting', 'question', or 'offtopic').  Any failure or
    ambiguous response defaults to 'eurocode' so the graph search always runs.
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
        if label.startswith("offtopic"):
            logger.info("Router: '%s' → offtopic", question[:60])
            return "offtopic"
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

    # ── Node: offtopic ─────────────────────────────────────────────
    def offtopic(state: AgentState) -> AgentState:
        return {
            **state,
            "answer": "Diese Frage liegt außerhalb meines Fachgebiets. "
                      "Ich kann nur Fragen zu Eurocodes und Tragwerksplanung beantworten.",
            "tools_used": [],
        }

    # ── Node: graph search ───────────────────────────────────────────
    def graph_search(state: AgentState) -> AgentState:
        question = state["question"]
        german_query = _translate_to_german(question, llm)
        tools_used: List[Dict[str, Any]] = []

        try:
            pages = querier.search_hybrid(
                question,                            # original → embedding paths
                limit=settings.hybrid_candidates_per_path,
                bm25_query=german_query,             # German → BM25 path
            )
            tools_used.append({
                "tool": "search_hybrid",
                "arguments": {"query": question, "bm25_query": german_query, "pages": len(pages)},
            })
        except Exception as e:
            logger.error("Hybrid search failed for '%s': %s", question[:80], e)
            pages = []
            tools_used.append({
                "tool": "search_hybrid",
                "arguments": {"query": question, "bm25_query": german_query},
                "error": str(e),
            })

        # Cross-encoder reranking
        from backend.app.modules.reranker import get_reranker
        reranker = get_reranker()
        if reranker and pages:
            try:
                pages = reranker.rerank(
                    question,
                    pages,
                    top_k=settings.reranker_top_k,
                    threshold=settings.reranker_threshold,
                )
                tools_used.append({
                    "tool": "rerank",
                    "arguments": {"pages_after": len(pages)},
                })
            except Exception as e:
                logger.warning("Reranker failed, using unranked pages: %s", e)

        # Graph enrichment
        extra_figures_raw: List[Dict[str, Any]] = []
        extra_formulas: List[Dict[str, Any]] = []
        try:
            pages, extra_figures_raw, extra_formulas = querier.enrich_with_graph(pages)
        except Exception as e:
            logger.debug("Graph enrichment failed: %s", e)

        return {
            **state,
            "keywords": keywords,
            "search_results": pages,
            "tools_used": tools_used,
            "_extra_figures_raw": extra_figures_raw,
            "_extra_formulas": extra_formulas,
        }

    # ── Node: build context ──────────────────────────────────────────
    def rank_context(state: AgentState) -> AgentState:
        pages = state.get("search_results", [])
        img_mapping: Dict[str, str] = {}
        img_counter: List[int] = [1]
        context = _format_pages_as_context(pages, img_mapping, img_counter)

        tools_used = list(state.get("tools_used", []))

        # Build extra_figures markdown from enrichment results
        extra_figures: List[str] = []
        extra_figures_raw = state.get("_extra_figures_raw", [])
        if extra_figures_raw:
            existing_paths: set = set()
            for v in img_mapping.values():
                if "](/" in v:
                    existing_paths.add(v.split("](")[1].rstrip(")"))
            seen_paths: set = set()
            for fig in extra_figures_raw:
                path = fig.get("image_path", "")
                caption = fig.get("caption", "") or fig.get("number", "")
                if path and path not in seen_paths and path not in existing_paths:
                    seen_paths.add(path)
                    extra_figures.append(f"![{caption}]({path})")

        tools_used.append({
            "tool": "rank_context",
            "arguments": {"top_items": context.count("["), "chars": len(context)},
        })

        return {**state, "ranked_context": context, "img_mapping": img_mapping, "extra_figures": extra_figures, "tools_used": tools_used}

    # ── Node: answer LLM ────────────────────────────────────────────
    def answer_llm(state: AgentState) -> AgentState:
        question = state["question"]
        context = state.get("ranked_context", "") or "(Keine relevanten Ergebnisse im Wissensgraphen gefunden.)"
        img_mapping = state.get("img_mapping", {})

        # Hard hallucination gate: skip LLM when context is empty or trivially short.
        _no_results_marker = "(Keine relevanten"
        if (context.startswith(_no_results_marker)
                or len(context) < settings.hallucination_min_context_chars):
            refusal = "Diese Information ist im bereitgestellten Kontext nicht vorhanden."
            logger.info("Hallucination gate triggered — context too short (%d chars), skipping LLM.", len(context))
            return {**state, "answer": refusal}

        extra_formulas = state.get("_extra_formulas", [])
        if extra_formulas:
            formula_lines = [
                f"- {f.get('section', '')}: $${f.get('latex') or f.get('unicode', '')}$$"
                for f in extra_formulas
                if f.get("latex") or f.get("unicode")
            ]
            if formula_lines:
                context += "\n\nRELATED FORMULAS:\n" + "\n".join(formula_lines)

        user_prompt = (
            "=== BEGIN RETRIEVED CONTEXT ===\n"
            f"{context}\n"
            "=== END RETRIEVED CONTEXT ===\n\n"
            f"QUESTION: {question}"   # ← question is the LAST thing the model reads
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
        if img_mapping:
            answer = _restore_image_placeholders(answer, img_mapping)

        # Append extra figures if any
        extra_figures = state.get("extra_figures", [])
        if extra_figures:
            answer += "\n\n---\n**Abbildungen:**\n" + "\n".join(extra_figures)

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
    graph.add_node("offtopic", offtopic)
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
            "offtopic": "offtopic",
            "eurocode": "graph_search",
        },
    )

    graph.add_edge("greet", END)
    graph.add_edge("offtopic", END)
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
            elif label.startswith("offtopic"):
                route = "offtopic"
        except Exception as e:
            logger.warning("Router call failed in stream, defaulting to eurocode: %s", e)

        # ── Off-topic path ─────────────────────────────────────────────
        if route == "offtopic":
            offtopic_msg = (
                "Diese Frage liegt außerhalb meines Fachgebiets. "
                "Ich kann nur Fragen zu Eurocodes und Tragwerksplanung beantworten."
            )
            yield {"type": "token", "content": offtopic_msg}
            yield {"type": "done", "answer": offtopic_msg, "tools_used": [], "route": "offtopic"}
            return

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
        # Step 2: translate to German for BM25 only, then run 4-path hybrid search
        yield {"type": "status", "step": "searching", "message": "Suche im Wissensgraphen…"}

        # bge-m3 is multilingual — embed the original question unchanged.
        # The German translation is passed separately and used ONLY by the BM25 path.
        german_query = await _atranslate_to_german(question, self.llm)

        tools_used: List[Dict[str, Any]] = []
        pages: List[Dict[str, Any]] = []
        search_debug: Dict[str, Any] = {}
        try:
            pages = self.querier.search_hybrid(
                question,                               # original → embedding paths
                limit=settings.hybrid_candidates_per_path,
                bm25_query=german_query,                # German → BM25 path
                _debug=search_debug,
            )
            tools_used.append({
                "tool": "search_hybrid",
                "arguments": {"query": question, "bm25_query": german_query, "pages": len(pages)},
            })
        except Exception as e:
            logger.error("Hybrid search failed in stream: %s", e)
            tools_used.append({
                "tool": "search_hybrid",
                "arguments": {"query": question, "bm25_query": german_query},
                "error": str(e),
            })

        # Snapshot page order before reranking for debug comparison
        def _pg_snap(p_list: List[Dict], score_key: str = "score") -> List[Dict]:
            return [
                {
                    "pg": p.get("page_number"),
                    "doc": (p.get("document") or "")[-50:],
                    "ch": (p.get("chapter") or "")[:40],
                    score_key: round(float(p.get(score_key) or 0), 3),
                }
                for p in p_list
            ]

        pages_before_rerank = _pg_snap(pages)

        # Step 2b: cross-encoder reranking
        yield {"type": "status", "step": "reranking", "message": "Bewerte Relevanz…"}
        from backend.app.modules.reranker import get_reranker
        reranker = get_reranker()
        if reranker and pages:
            try:
                pages = reranker.rerank(
                    question,
                    pages,
                    top_k=settings.reranker_top_k,
                    threshold=settings.reranker_threshold,
                )
                tools_used.append({
                    "tool": "rerank",
                    "arguments": {"pages_after": len(pages)},
                })
            except Exception as e:
                logger.warning("Reranker failed, using unranked pages: %s", e)

        pages_after_rerank = _pg_snap(pages, "rerank_score")

        # Step 2c: graph enrichment (adjacent pages, figures, formulas)
        pages_before_enrich = len(pages)
        extra_figures_raw: List[Dict[str, Any]] = []
        extra_formulas: List[Dict[str, Any]] = []
        try:
            pages, extra_figures_raw, extra_formulas = self.querier.enrich_with_graph(pages)
            if extra_figures_raw or extra_formulas:
                tools_used.append({
                    "tool": "enrich_with_graph",
                    "arguments": {
                        "figures": len(extra_figures_raw),
                        "formulas": len(extra_formulas),
                    },
                })
        except Exception as e:
            logger.debug("Graph enrichment failed: %s", e)

        adjacent_added = _pg_snap(pages[pages_before_enrich:])

        # Step 3: apply context budget, then format pages as context
        yield {"type": "status", "step": "ranking", "message": "Bereite Kontext vor…"}
        budget = settings.total_context_budget
        pages_for_context = pages
        if budget > 0:
            trimmed: List[Dict[str, Any]] = []
            used_chars = 0
            for p in pages:
                plen = len(p.get("content", ""))
                if used_chars + plen > budget and trimmed:
                    break
                trimmed.append(p)
                used_chars += plen
            pages_for_context = trimmed

        img_mapping: Dict[str, str] = {}
        img_counter: List[int] = [1]
        context = _format_pages_as_context(pages_for_context, img_mapping, img_counter)

        # Build extra_figures markdown from enrichment results
        extra_figures: List[str] = []
        if extra_figures_raw:
            existing_paths: set = set()
            for v in img_mapping.values():
                if "](/" in v:
                    existing_paths.add(v.split("](")[1].rstrip(")"))
            seen_paths: set = set()
            for fig in extra_figures_raw:
                path = fig.get("image_path", "")
                caption = fig.get("caption", "") or fig.get("number", "")
                if path and path not in seen_paths and path not in existing_paths:
                    seen_paths.add(path)
                    extra_figures.append(f"![{caption}]({path})")

        tools_used.append({
            "tool": "build_context",
            "arguments": {"pages": len(pages_for_context), "chars": len(context)},
        })

        # Step 4: stream LLM answer
        yield {"type": "status", "step": "answering", "message": "Generiere Antwort…"}

        # Hard hallucination gate: if no meaningful context, skip LLM entirely.
        _no_results_marker = "(Keine relevanten"
        if (context.startswith(_no_results_marker)
                or len(context) < settings.hallucination_min_context_chars):
            refusal = "Diese Information ist im bereitgestellten Kontext nicht vorhanden."
            logger.info("Hallucination gate triggered — context too short (%d chars), skipping LLM.", len(context))
            _write_debug({
                "ts": datetime.utcnow().isoformat(),
                "question": question,
                "german_query": german_query,
                "search": search_debug,
                "rerank": {"before": pages_before_rerank, "after": [], "gate": "hallucination_gate"},
                "enrichment": {"adjacent_added": adjacent_added, "figures": 0, "formulas": 0},
                "budget": {"pages_input": len(pages), "pages_used": 0, "budget_chars": budget, "context_chars": 0},
                "context": "",
                "answer": refusal,
            })
            yield {"type": "token", "content": refusal}
            yield {
                "type": "done",
                "answer": refusal,
                "tools_used": tools_used,
                "route": "eurocode",
                "sources": [],
            }
            return

        if extra_formulas:
            formula_lines = [
                f"- {f.get('section', '')}: $${f.get('latex') or f.get('unicode', '')}$$"
                for f in extra_formulas
                if f.get("latex") or f.get("unicode")
            ]
            if formula_lines:
                context += "\n\nRELATED FORMULAS:\n" + "\n".join(formula_lines)

        user_prompt = (
            "=== BEGIN RETRIEVED CONTEXT ===\n"
            f"{context}\n"
            "=== END RETRIEVED CONTEXT ===\n\n"
            f"QUESTION: {question}"   # ← question is the LAST thing the model reads
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

        # Step 5: post-process — restore image placeholders + fix LaTeX + append extra figures
        if img_mapping:
            full_answer = _restore_image_placeholders(full_answer, img_mapping)
        full_answer = _postprocess_latex(full_answer)

        # Append extra figures if any
        if extra_figures:
            full_answer += "\n\n---\n**Abbildungen:**\n" + "\n".join(extra_figures)

        # Build source list — one entry per page, same order and count as context.
        # Use `is not None` so page 0 is not excluded by falsy truthiness check.
        sources = [
            {
                "page_number": p.get("page_number"),
                "document":    p.get("document") or "",
                "chapter":     p.get("chapter", ""),
                "preview":     (p.get("content") or "")[:600].strip(),
            }
            for p in pages_for_context
            if p.get("page_number") is not None and p.get("document")
        ]

        # ── Single comprehensive debug entry ──────────────────────────
        # Captures every step of the pipeline so accuracy problems can be
        # diagnosed by reading debug_raw.jsonl without touching logs.
        _write_debug({
            "ts": datetime.utcnow().isoformat(),
            "question": question,
            "german_query": german_query,
            "search": search_debug,
            "rerank": {
                "before_n": len(pages_before_rerank),
                "before": pages_before_rerank,
                "after_n": len(pages_after_rerank),
                "after": pages_after_rerank,
                "dropped": [
                    p for p in pages_before_rerank
                    if (p["pg"], p["doc"]) not in
                    {(q["pg"], q["doc"]) for q in pages_after_rerank}
                ],
            },
            "enrichment": {
                "pages_before": pages_before_enrich,
                "adjacent_added": adjacent_added,
                "figures": len(extra_figures_raw),
                "formulas": len(extra_formulas),
                "pages_after": len(pages),
            },
            "budget": {
                "pages_input": len(pages),
                "pages_used": len(pages_for_context),
                "dropped_by_budget": len(pages) - len(pages_for_context),
                "budget_chars": budget,
                "context_chars": len(context),
            },
            "user_prompt": user_prompt,
            "answer": full_answer,
        })

        yield {"type": "done", "answer": full_answer, "tools_used": tools_used, "route": "eurocode", "sources": sources}

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
