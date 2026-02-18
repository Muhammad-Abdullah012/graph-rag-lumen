"""Eurocode Agent - LangGraph ReAct agent with Graph-RAG tools"""
import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from backend.app.modules.graph_querier import get_graph_querier, GraphQuerier
from config.settings import settings

logger = logging.getLogger(__name__)

# ================================================================== #
#  System prompt
# ================================================================== #
SYSTEM_MESSAGE = (
    "You are an expert Eurocode civil-engineering assistant backed by a Graph-RAG "
    "knowledge graph. The graph contains Documents, Chapters, Pages, Sections, "
    "Tables, Figures, Formulas, and Concepts — all linked with structural and "
    "semantic relationships.\n\n"
    "ALWAYS use your tools to query the knowledge graph before answering — never "
    "guess or invent data.\n\n"
    "Available tool strategies:\n"
    "• Use `search` as the primary broad search — it queries sections, concepts, "
    "  tables, figures, formulas, and performs semantic vector search.\n"
    "• Use `lookup_concept` for specific engineering terms or symbols "
    "  (γf, Ed, 'Einwirkung', 'limit state', etc.).\n"
    "• Use `search_formulas` when the user asks about a formula or equation.\n"
    "• Use `search_tables` when the user asks about table data.\n"
    "• Use `list_documents` or `list_chapters` for structural navigation.\n\n"
    "Include exact concept names, definitions, formulas, and section/page citations "
    "in your answer. If the question is German, answer in German. If English, answer "
    "in English. Be precise and cite the document and section.\n\n"
    "IMPORTANT: The knowledge graph data is primarily in GERMAN. When the user asks "
    "in English, translate search terms to German before calling tools. Examples:\n"
    "- 'partial safety factor' → search for 'Teilsicherheitsbeiwert'\n"
    "- 'action' / 'load' → search for 'Einwirkung'\n"
    "- 'resistance' → 'Widerstand'\n"
    "- 'abbreviation' → search the abbreviation directly (EQU, SLS, ULS)\n\n"
    "When looking up Greek symbols like γf, γG, γQ, use exact Unicode characters."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MESSAGE),
    MessagesPlaceholder(variable_name="messages"),
])


# ================================================================== #
#  Tool definitions using @tool decorator
# ================================================================== #
def _setup_tools(querier: GraphQuerier):
    """Create LangChain tools that call the GraphQuerier methods."""

    @tool
    def search(query: str) -> str:
        """Search across the entire Graph-RAG knowledge graph: sections, concepts,
        tables, figures, formulas, and perform semantic vector similarity.
        This is the PRIMARY search tool — use it for any question.
        Works with German and English queries.
        Input: a natural-language query or keyword."""
        results = querier.general_search(query)
        if not results:
            return f"No results found for '{query}'."
        return json.dumps(results, ensure_ascii=False, default=str)

    @tool
    def lookup_concept(name: str) -> str:
        """Look up a specific concept, symbol, abbreviation or definition by name.
        Concepts include Eurocode symbols (γf, Ed, Fd), engineering terms
        (Einwirkung, Tragfähigkeit), and abbreviations (EQU, SLS, ULS).
        Returns the concept description, related concepts, and all sections
        that mention it.
        Input: exact concept/symbol name."""
        combined: List[Dict[str, Any]] = []
        seen: set = set()

        def _add(items):
            for item in (items or []):
                key = item.get("concept") or item.get("symbol") or item.get("term", "")
                if key not in seen:
                    seen.add(key)
                    combined.append(item)

        _add(querier.lookup_concept(name))
        _add(querier.lookup_symbol(name))
        _add(querier.search_concepts(name))

        if not combined:
            return f"No concept found for '{name}'."
        return json.dumps(combined, ensure_ascii=False, default=str)

    @tool
    def search_formulas(keyword: str) -> str:
        """Search for formulas / equations in the knowledge graph.
        Returns LaTeX expressions with the section and document they appear in.
        Input: keyword (e.g. 'AEd', 'Erdbeben', 'Formel', 'combination')."""
        results = querier.search_formulas(keyword)
        if not results:
            results = querier.list_formulas()
        if not results:
            return f"No formulas found for '{keyword}'."
        return json.dumps(results, ensure_ascii=False, default=str)

    @tool
    def search_tables(keyword: str) -> str:
        """Search for tables in the knowledge graph by caption or content.
        Input: keyword describing the table."""
        results = querier.search_tables(keyword)
        if not results:
            return f"No tables found for '{keyword}'."
        return json.dumps(results, ensure_ascii=False, default=str)

    @tool
    def list_documents() -> str:
        """List all documents in the knowledge graph with their types and section counts.
        No input needed."""
        results = querier.list_documents()
        if not results:
            return "No documents found in the knowledge graph."
        return json.dumps(results, ensure_ascii=False, default=str)

    @tool
    def list_chapters(document_keyword: str = "") -> str:
        """List chapters in the knowledge graph, optionally filtered by document.
        Input: optional document name keyword (leave empty for all)."""
        results = querier.list_chapters(document_keyword or None)
        if not results:
            return f"No chapters found for '{document_keyword}'."
        return json.dumps(results, ensure_ascii=False, default=str)

    return [
        search,
        lookup_concept,
        search_formulas,
        search_tables,
        list_documents,
        list_chapters,
    ]


# ================================================================== #
#  Agent class
# ================================================================== #
class EurocodeAgent:
    """
    LangGraph ReAct agent that uses native tool calling to query
    the Eurocode knowledge graph and produce grounded answers.
    """

    def __init__(self):
        self.querier = get_graph_querier()
        self.tools = _setup_tools(self.querier)

        # Initialize ChatOllama (LangChain wrapper for Ollama with tool calling support)
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_llm_model,
            temperature=0.1,
        )

        # Create the LangGraph ReAct agent
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=prompt,
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
