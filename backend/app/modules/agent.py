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
Answer the user's question using ONLY what is in the CONTEXT.

FORMULAS:
- In the CONTEXT, block formulas are wrapped in $$...$$ and inline variables in $...$.
- When the user asks for a formula, find the relevant $$...$$ block(s) and copy the EXACT characters between and including the $$ markers — letter for letter, symbol for symbol.
- The formula in your answer MUST be identical to the formula in the CONTEXT. Do not change notation, subscripts, operators, or structure in any way.
- Do NOT rewrite, simplify, rearrange, or paraphrase any formula — not even slightly.
- Do NOT write a formula that is not present in the CONTEXT.

TABLES:
- The CONTEXT contains markdown tables using pipe syntax (| col1 | col2 |).
- ALWAYS include the relevant table in your answer by copying the pipe-delimited markdown table EXACTLY as it appears in the CONTEXT.
- NEVER summarize, paraphrase, or convert a table to bullet points or prose. ALWAYS output it as a markdown table.
- NEVER say "see Table X.Y", "refer to the table", or "as shown in the table" — the user cannot see the table unless you copy it.
- You may filter to only the relevant rows, but ALWAYS keep the header row and the separator row (| --- | --- |).
- Example — if the CONTEXT has:
  | Kerbfall | Beschreibung | Anforderungen |
  | --- | --- | --- |
  | 71 | Detail 1 | R ≥ 150 |
  | 80 | Detail 2 | l ≤ 50mm |
  then your answer MUST include:
  | Kerbfall | Beschreibung | Anforderungen |
  | --- | --- | --- |
  | 71 | Detail 1 | R ≥ 150 |

LANGUAGE:
- Reply in the same language as the user's question.

IF NOT IN CONTEXT:
- If the answer is not in the CONTEXT, you MUST write ONLY this exact sentence: "Diese Information ist im bereitgestellten Kontext nicht vorhanden."
- Do NOT write anything else. Do NOT add general knowledge, suggestions, or explanations from your training data.
- Do NOT say "generally speaking", "in Eurocode...", "typically...", or anything similar.
- Silence is better than a wrong answer.
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
    "  design value → Bemessungswert\n"
    "  characteristic value → charakteristischer Wert\n"
    "  shear modulus → Schubmodul\n"
    "  elastic modulus / modulus of elasticity → Elastizitätsmodul\n"
    "  yield strength → Streckgrenze\n"
    "  partial factor / partial safety factor → Teilsicherheitsbeiwert\n"
    "  load combination → Lastkombination\n"
    "  dynamic amplification factor → Schwingbeiwert\n"
    "  steel → Stahl\n"
    "  concrete → Beton\n"
    "  bridge → Brücke\n"
    "  material constant → Materialkonstante\n"
    "  Poisson's ratio → Querdehnzahl\n"
    "  coefficient of thermal expansion → Wärmeausdehnungskoeffizient\n"
    "  density → Wichte / Rohdichte\n"
    "  section → Abschnitt"
)

_KEYWORD_SYSTEM = (
    "You are a search query optimizer for a Eurocode structural engineering knowledge base. "
    "Extract 3-6 key technical search terms from the query below.\n\n"
    "Rules:\n"
    "- Include specific Eurocode terms, load types, material names, norm numbers, "
    "  numeric values (e.g. 71, 8.3), and abbreviations (e.g. EC3, ψ0, γQ)\n"
    "- Keep the original German spelling exactly — do NOT translate or modify terms\n"
    "- Exclude question words (welche, was, wie), prepositions, articles, "
    "  conjunctions, and generic verbs (berücksichtigen, bestimmen, etc.)\n"
    "- Return ONLY a space-separated list of terms — no explanation, no punctuation, no quotes\n\n"
    "Examples:\n"
    "  Query: 'Welche Kombinationswerte sind für Temperatur bei einer Straßenbrücke zu berücksichtigen?'\n"
    "  Output: Kombinationswerte Temperatur Straßenbrücke\n\n"
    "  Query: 'Kerbfall 71 Kerbdetail Kategorie'\n"
    "  Output: Kerbfall 71 Kerbdetail Kategorie\n\n"
    "  Query: 'Welche Lastmodelle gibt es für Eisenbahnbrücken nach EN 1991-2?'\n"
    "  Output: Lastmodelle Eisenbahnbrücken EN 1991-2\n\n"
    "  Query: 'Was ist der Teilsicherheitsbeiwert γQ für veränderliche Einwirkungen?'\n"
    "  Output: Teilsicherheitsbeiwert γQ veränderliche Einwirkungen"
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


def _extract_keywords(question: str, llm: Any) -> str:
    """Extract key technical search terms from *question* using the LLM.

    Returns a space-separated string of keywords (e.g. "Kombinationswerte
    Temperatur Straßenbrücke").  Returns an empty string on failure so the
    caller can fall back to the static stopword filter in GraphQuerier.
    """
    try:
        resp = llm.invoke([
            SystemMessage(content=_KEYWORD_SYSTEM),
            HumanMessage(content=question.strip()),
        ])
        keywords = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        # Sanity: reject multi-line responses (model returned an explanation)
        keywords = keywords.splitlines()[0].strip() if keywords else ""
        if keywords:
            logger.info("Keywords extracted: '%s' → '%s'", question[:60], keywords[:80])
            return keywords
    except Exception as e:
        logger.warning("Keyword extraction failed, falling back to stopword filter: %s", e)
    return ""


async def _aextract_keywords(question: str, llm: Any) -> str:
    """Async version of _extract_keywords."""
    try:
        resp = await llm.ainvoke([
            SystemMessage(content=_KEYWORD_SYSTEM),
            HumanMessage(content=question.strip()),
        ])
        keywords = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        keywords = keywords.splitlines()[0].strip() if keywords else ""
        if keywords:
            logger.info("Keywords extracted: '%s' → '%s'", question[:60], keywords[:80])
            return keywords
    except Exception as e:
        logger.warning("Keyword extraction failed, falling back to stopword filter: %s", e)
    return ""


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
        keywords = _extract_keywords(question, llm)
        tools_used: List[Dict[str, Any]] = []

        try:
            pages = querier.search_hybrid(
                question,
                limit=settings.hybrid_candidates_per_path,
                keywords=keywords,
            )
            tools_used.append({
                "tool": "search_hybrid",
                "arguments": {"query": question, "keywords": keywords, "pages": len(pages)},
            })
        except Exception as e:
            logger.error("Hybrid search failed for '%s': %s", question[:80], e)
            pages = []
            tools_used.append({
                "tool": "search_hybrid",
                "arguments": {"query": question, "keywords": keywords},
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

        user_prompt = (
            f"CONTEXT:\n{context}\n\n"
            "REMINDER:\n"
            "- TABLES: If the CONTEXT contains a markdown table (lines with | ), "
            "you MUST copy it into your answer as a markdown table. "
            "Do NOT convert tables to bullet points or prose.\n"
            "- FORMULAS: If the CONTEXT contains formulas in $$...$$ or $...$, "
            "copy them EXACTLY into your answer — do not rewrite or omit them.\n\n"
            f"QUESTION: {question}"
        )

        # Append enriched formulas from graph enrichment
        extra_formulas = state.get("_extra_formulas", [])
        if extra_formulas:
            formula_lines = [
                f"- {f.get('section', '')}: $${f.get('latex') or f.get('unicode', '')}$$"
                for f in extra_formulas
                if f.get("latex") or f.get("unicode")
            ]
            if formula_lines:
                user_prompt += "\n\nRELATED FORMULAS:\n" + "\n".join(formula_lines)

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
        # Step 2: translate → extract keywords → 3-path hybrid search
        yield {"type": "status", "step": "searching", "message": "Suche im Wissensgraphen…"}

        keywords = await _aextract_keywords(question, self.llm)

        tools_used: List[Dict[str, Any]] = []
        pages: List[Dict[str, Any]] = []
        try:
            pages = self.querier.search_hybrid(
                question,
                limit=settings.hybrid_candidates_per_path,
                keywords=keywords,
            )
            tools_used.append({
                "tool": "search_hybrid",
                "arguments": {"query": question, "keywords": keywords, "pages": len(pages)},
            })
        except Exception as e:
            logger.error("Hybrid search failed in stream: %s", e)
            tools_used.append({
                "tool": "search_hybrid",
                "arguments": {"query": question, "keywords": keywords},
                "error": str(e),
            })

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

        # Step 2c: graph enrichment (adjacent pages, figures, formulas)
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

        # Step 3: format pages as context (replace images with placeholders)
        yield {"type": "status", "step": "ranking", "message": "Bereite Kontext vor…"}
        img_mapping: Dict[str, str] = {}
        img_counter: List[int] = [1]
        context = _format_pages_as_context(pages, img_mapping, img_counter)

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
            "arguments": {"pages": len(pages), "chars": len(context)},
        })

        # ── Debug log ─────────────────────────────────────────────────
        _write_debug({
            "ts": datetime.utcnow().isoformat(),
            "question": question,
            "pages_found": len(pages),
            "page_titles": [f"p{p.get('page_number','?')} {p.get('chapter','')}" for p in pages],
            "context": context,
        })

        # Step 4: stream LLM answer
        yield {"type": "status", "step": "answering", "message": "Generiere Antwort…"}

        user_prompt = (
            f"CONTEXT:\n{context}\n\n"
            "REMINDER:\n"
            "- TABLES: If the CONTEXT contains a markdown table (lines with | ), "
            "you MUST copy it into your answer as a markdown table. "
            "Do NOT convert tables to bullet points or prose.\n"
            "- FORMULAS: If the CONTEXT contains formulas in $$...$$ or $...$, "
            "copy them EXACTLY into your answer — do not rewrite or omit them.\n\n"
            f"QUESTION: {question}"
        )

        # Append enriched formulas from graph enrichment
        if extra_formulas:
            formula_lines = [
                f"- {f.get('section', '')}: $${f.get('latex') or f.get('unicode', '')}$$"
                for f in extra_formulas
                if f.get("latex") or f.get("unicode")
            ]
            if formula_lines:
                user_prompt += "\n\nRELATED FORMULAS:\n" + "\n".join(formula_lines)

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
            for p in pages
            if p.get("page_number") is not None and p.get("document")
        ]

        # ── Debug log: final answer (for hallucination checking) ──────
        _write_debug({
            "ts": datetime.utcnow().isoformat(),
            "question": question,
            "final_answer": full_answer,
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
