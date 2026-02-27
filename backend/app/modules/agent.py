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

ANSWER_SYSTEM = """You are a Eurocode structural engineering assistant.
You are given CONTEXT pages retrieved from official Eurocode documents.
Answer the user's question using ONLY what is in the CONTEXT.

FORMULAS:
- In the CONTEXT, block formulas are wrapped in $$...$$ and inline variables in $...$.
- When the user asks for a formula, find the relevant $$...$$ block(s) and copy the EXACT characters between and including the $$ markers — letter for letter, symbol for symbol.
- The formula in your answer MUST be identical to the formula in the CONTEXT. Do not change notation, subscripts, operators, or structure in any way.
- Do NOT rewrite, simplify, rearrange, or paraphrase any formula — not even slightly.
- Do NOT write a formula that is not present in the CONTEXT.

IMAGES:
- Images appear in the CONTEXT as ![caption](url). Copy them VERBATIM if relevant.
- Do NOT invent image URLs.

LANGUAGE:
- Reply in the same language as the user's question.

IF NOT IN CONTEXT:
- If the answer is not in the CONTEXT, write only: "Diese Information ist im bereitgestellten Kontext nicht vorhanden."
- Do NOT add values or explanations from your training data.
"""


# ================================================================== #
#  Context formatting
# ================================================================== #


def _format_pages_as_context(pages: List[Dict[str, Any]]) -> str:
    """Format a list of pages as context for the LLM.

    Each page entry already contains the full OCR text of that page —
    formulas, tables, and figures are all inline in the page content.
    No separate lookup is needed.
    """
    if not pages:
        return "(Keine relevanten Seiten im Wissensgraphen gefunden.)"

    parts: List[str] = []
    for i, page in enumerate(pages, 1):
        page_num = page.get("page_number", "?")
        chapter  = page.get("chapter", "")
        doc      = page.get("document", "")
        content  = page.get("content", "")

        ref      = f" (Dokument: {doc})" if doc else ""
        chap_str = f" | {chapter}" if chapter else ""
        parts.append(f"[{i}] Seite {page_num}{chap_str}{ref}\n{content}")

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
        search_query = _translate_to_german(question, llm)
        tools_used: List[Dict[str, Any]] = []

        try:
            pages = querier.search_pages(search_query, limit=8)
            tools_used.append({"tool": "search_pages", "arguments": {"query": search_query}})
        except Exception as e:
            logger.error("Page search failed for '%s': %s", search_query[:80], e)
            pages = []
            tools_used.append({
                "tool": "search_pages",
                "arguments": {"query": search_query},
                "error": str(e),
            })

        return {
            **state,
            "search_query": search_query,
            "search_results": pages,
            "tools_used": tools_used,
        }

    # ── Node: build context ──────────────────────────────────────────
    def rank_context(state: AgentState) -> AgentState:
        pages = state.get("search_results", [])
        context = _format_pages_as_context(pages)

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

        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"

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
        # Step 2: translate + search top 8 pages
        yield {"type": "status", "step": "searching", "message": "Suche im Wissensgraphen…"}
        search_query = await _atranslate_to_german(question, self.llm)

        tools_used: List[Dict[str, Any]] = []
        pages: List[Dict[str, Any]] = []
        try:
            pages = self.querier.search_pages(search_query, limit=8)
            tools_used.append({"tool": "search_pages", "arguments": {"query": search_query, "pages": len(pages)}})
        except Exception as e:
            logger.error("Page search failed in stream: %s", e)
            tools_used.append({"tool": "search_pages", "arguments": {"query": search_query}, "error": str(e)})

        # Step 3: format pages as context
        yield {"type": "status", "step": "ranking", "message": "Bereite Kontext vor…"}
        context = _format_pages_as_context(pages)
        tools_used.append({
            "tool": "build_context",
            "arguments": {"pages": len(pages), "chars": len(context)},
        })

        # ── Debug log ─────────────────────────────────────────────────
        _write_debug({
            "ts": datetime.utcnow().isoformat(),
            "question": question,
            "search_query": search_query,
            "pages_found": len(pages),
            "page_titles": [f"p{p.get('page_number','?')} {p.get('chapter','')}" for p in pages],
            "context": context,
        })

        # Step 4: stream LLM answer
        yield {"type": "status", "step": "answering", "message": "Generiere Antwort…"}

        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"

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

        # ── Debug log: final answer (for hallucination checking) ──────
        _write_debug({
            "ts": datetime.utcnow().isoformat(),
            "question": question,
            "final_answer": full_answer,
        })

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
