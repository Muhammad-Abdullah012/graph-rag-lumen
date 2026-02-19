"""Eurocode Agent - LangChain agent with Graph-RAG tools"""
import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from backend.app.modules.graph_querier import get_graph_querier, GraphQuerier
from config.settings import settings

logger = logging.getLogger(__name__)

# ================================================================== #
#  System prompt
# ================================================================== #
SYSTEM_MESSAGE = (
    "You are a Eurocode structural engineering expert backed by a Graph-RAG "
    "knowledge graph. The graph contains Documents, Chapters, Pages, Sections, "
    "Tables, Figures, Formulas, and Concepts — linked by structural and semantic "
    "relationships.\n\n"
    "CRITICAL RULE — LANGUAGE:\n"
    "Detect the language of the user's question. "
    "If the user writes in English, you MUST answer in English. "
    "If the user writes in German, you MUST answer in German. "
    "The search results from the knowledge graph are in German — that is fine, "
    "but your final answer MUST match the user's language.\n\n"
    "CRITICAL RULE — ANSWER QUALITY:\n"
    "1. Read the search results carefully and extract ONLY the information that "
    "   directly answers the user's question.\n"
    "2. Do NOT dump or list raw search results. Synthesize a clear, direct answer.\n"
    "3. If the question asks 'which types' or 'what are', give a specific list.\n"
    "4. If the question asks 'how', explain the procedure step by step.\n"
    "5. Ignore search results that are not relevant to the question.\n"
    "6. Cite the document name, section number, and page when possible.\n"
    "7. Render formulas in LaTeX format.\n\n"
    "TOOL USAGE:\n"
    "1. Always use `search` first for every question.\n"
    "2. Use `lookup` for specific terms, symbols, or abbreviations.\n"
    "3. Use `navigate` only for document structure questions.\n"
    "4. If the first search does not answer the question, try a different query.\n\n"
    "SEARCH TIPS — the graph data is in GERMAN, so translate search terms:\n"
    "- scope / application → Anwendungsbereich\n"
    "- bridge → Brücke\n"
    "- excluded / exclusion → ausgeschlossen / Ausschluss\n"
    "- partial safety factor → Teilsicherheitsbeiwert\n"
    "- action / load → Einwirkung\n"
    "- resistance → Widerstand\n"
    "- Use Greek symbols as Unicode: γf, γG, γQ"
)


# ================================================================== #
#  Tool definitions using @tool decorator
# ================================================================== #
def _setup_tools(querier: GraphQuerier):
    """Create 3 essential LangChain tools for the Graph-RAG agent."""

    @tool
    def search(query: str) -> str:
        """Search the entire Eurocode knowledge graph: sections, concepts,
        tables, figures, formulas, and semantic vector search.
        Primary tool — use for every question.
        Input: search term or phrase. Use GERMAN terms for best results
        (e.g. 'Anwendungsbereich' not 'scope')."""
        results = querier.general_search(query)
        if not results:
            return f"No results found for '{query}'."
        # Build focused output — prioritize sections with content
        # TODO: Need to update it properly
        output: Dict[str, Any] = {}
        for key, items in results.items():
            if isinstance(items, list):
                # Limit items per category
                limit = 5 if key in ("sections", "semantic") else 3
                output[key] = items[:limit]
            else:
                output[key] = items
        return json.dumps(output, ensure_ascii=False, default=str)

    @tool
    def lookup(name: str) -> str:
        """Look up a specific concept, symbol, or abbreviation.
        Returns description, related concepts, and all sections mentioning it.
        Input: exact name (e.g. 'γf', 'Einwirkung', 'EQU', 'Teilsicherheitsbeiwert')."""
        combined: List[Dict[str, Any]] = []
        seen: set = set()

        def _add(items):
            for item in (items or []):
                key = item.get("concept") or item.get("term") or item.get("section", "")
                if key and key not in seen:
                    seen.add(key)
                    combined.append(item)

        _add(querier.lookup_concept(name))
        _add(querier.search_concepts(name, limit=5))
        _add(querier.get_concept_sections(name))

        if not combined:
            return f"No concept found for '{name}'."
        return json.dumps(combined[:15], ensure_ascii=False, default=str)

    @tool
    def navigate(document_keyword: str = "") -> str:
        """List documents and chapters in the knowledge graph.
        Optionally filter by document name.
        Input: optional document name keyword (empty for all)."""
        docs = querier.list_documents()
        chapters = querier.list_chapters(document_keyword or None)
        result = {
            "documents": docs[:20] if docs else [],
            "chapters": chapters[:30] if chapters else [],
        }
        if not docs and not chapters:
            return "No documents found in the knowledge graph."
        return json.dumps(result, ensure_ascii=False, default=str)

    return [search, lookup, navigate]


# ================================================================== #
#  Agent class
# ================================================================== #
class EurocodeAgent:
    """
    LangChain agent that uses tool calling to query
    the Eurocode knowledge graph and produce grounded answers.
    """

    def __init__(self):
        self.querier = get_graph_querier()
        self.tools = _setup_tools(self.querier)

        # Initialize ChatOllama (LangChain wrapper for Ollama with tool calling support)
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_llm_model,
            temperature=0,
        )

        # Create the agent
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SYSTEM_MESSAGE,
        )

        logger.info(
            "EurocodeAgent initialized with %d tools, model=%s",
            len(self.tools),
            settings.ollama_llm_model,
        )

    async def astream_answer(self, question: str):
        """
        Stream the agent's answer for a question.
        Yields text chunks and tool-call events as SSE data lines.
        """
        agent_input = {"messages": [("human", question)]}
        tool_names_seen = set()

        async for event in self.agent.astream_events(
            agent_input,
            version="v2",
        ):
            kind = event.get("event", "")

            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield f"data: {chunk.content}\n\n"

                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        if isinstance(tc, dict) and "name" in tc:
                            name = tc["name"]
                            if name not in tool_names_seen:
                                yield f"data: TOOL_USED:{name}\n\n"
                                tool_names_seen.add(name)

        yield "data: [DONE]\n\n"

    async def aanswer(self, question: str) -> Dict[str, Any]:
        """
        Non-streaming: invoke the agent and return the final answer
        plus metadata about which tools were used.
        """
        agent_input = {"messages": [("human", question)]}
        result = await self.agent.ainvoke(agent_input)

        # Extract the final AI message
        messages = result.get("messages", [])
        answer = ""
        tools_used = []

        for msg in messages:
            # Collect tool calls from AI messages
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_used.append({
                        "tool": tc.get("name", "unknown"),
                        "arguments": tc.get("args", {}),
                    })
            # The last AI message with content is the final answer
            if hasattr(msg, "content") and msg.content and msg.type == "ai":
                answer = msg.content

        return {
            "answer": answer,
            "tools_used": tools_used,
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
