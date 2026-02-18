"""Eurocode Agent - LangGraph ReAct agent with graph-query tools"""
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
    "You are an expert Eurocode civil-engineering assistant. "
    "You have access to a knowledge graph containing Eurocode symbols, formulas, "
    "abbreviations, definitions, units and references. "
    "ALWAYS use your tools to query the knowledge graph before answering — never guess or invent data. "
    "If the user asks about symbols, formulas, or definitions, call the appropriate tool first. "
    "Include exact symbol names, definitions, and formulas from the tool results in your answer. "
    "If the question is in German, answer in German. If in English, answer in English. "
    "Be precise and cite the document/section where information was found. "
    "If no relevant data is found after querying, say so honestly. "
    "\n\n"
    "IMPORTANT: The knowledge graph data is in GERMAN. When the user asks in English, "
    "you MUST translate the search terms to German before calling tools. Examples:\n"
    "- 'partial safety factor' → search for 'Teilsicherheitsbeiwert'\n"
    "- 'action' / 'load' → search for 'Einwirkung'\n"
    "- 'resistance' → search for 'Widerstand'\n"
    "- 'force' → search for 'Kraft'\n"
    "- 'unit weight' → search for 'Wichte'\n"
    "- 'formula' → search for 'Formel'\n"
    "- 'abbreviation' → search for the abbreviation directly (EQU, SLS, ULS etc.)\n"
    "\n"
    "When looking up specific Greek symbols like γf, γG, γQ etc., use the exact symbol "
    "characters as they appear. The graph stores them as Unicode: γ (gamma), φ (phi), etc."
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
    def lookup_symbols(query: str) -> str:
        """Look up or search for Eurocode symbols.
        Handles all symbol queries:
        - Exact lookup by name (e.g. 'γf', 'Ed', 'Fd')
        - Keyword search by concept (e.g. 'Teilsicherheitsbeiwert', 'partial safety factor')
        - Section-based listing (e.g. 'Griechische Buchstaben', 'Latin symbols')
        Tries exact match first, then full-text search, then section search.
        Input: symbol name, concept keyword, or section name."""
        combined: List[Dict[str, Any]] = []
        seen_keys: set = set()

        def _add(items):
            for item in (items or []):
                key = (item.get("symbol") or item.get("term") or "", item.get("section", ""))
                if key not in seen_keys:
                    seen_keys.add(key)
                    combined.append(item)

        # 1. Exact name lookup
        _add(querier.lookup_symbol(query))
        # 2. Keyword search (name / definition)
        _add(querier.search_symbols(query))
        # 3. Section-based listing
        _add(querier.get_symbols_in_section(query))

        if not combined:
            return f"No symbols found for '{query}'."
        return json.dumps(combined, ensure_ascii=False, default=str)

    @tool
    def lookup_formula(formula_name: str) -> str:
        """Look up a specific formula by name, including the expression and its variable definitions.
        Use when the user asks about a specific formula like 'AEd formula' or 'Erdbeben formula'.
        Input: keyword matching the formula name."""
        results = querier.get_formula(formula_name)
        if not results:
            return f"No formula found matching '{formula_name}'."
        return json.dumps(results, ensure_ascii=False, default=str)

    @tool
    def list_formulas() -> str:
        """List all formulas stored in the knowledge graph.
        Use when the user asks 'Show all formulas' or 'What formulas are available?'.
        No input needed."""
        results = querier.list_formulas()
        if not results:
            return "No formulas found in the knowledge graph."
        return json.dumps(results, ensure_ascii=False, default=str)

    @tool
    def lookup_abbreviations(abbreviation: str) -> str:
        """Look up the meaning of an abbreviation (e.g. 'EQU', 'SLS', 'ULS', 'GEO-2', 'STR').
        Input: the abbreviation."""
        results = querier.lookup_abbreviation(abbreviation)
        if not results:
            return f"No abbreviation found matching '{abbreviation}'."
        return json.dumps(results, ensure_ascii=False, default=str)

    @tool
    def get_unit(quantity_keyword: str) -> str:
        """Get the recommended Eurocode unit for a physical quantity
        (e.g. 'Kraft', 'Moment', 'Spannung', 'Dichte').
        Input: quantity keyword."""
        results = querier.get_unit(quantity_keyword)
        if not results:
            return f"No unit found for quantity '{quantity_keyword}'."
        return json.dumps(results, ensure_ascii=False, default=str)

    @tool
    def search(query: str) -> str:
        """Search across ALL knowledge graph node types: symbols, abbreviations, definitions,
        units, formulas, paragraphs, tables, images, chapters, sections, and document content blocks.
        Also performs semantic (vector) similarity search over embedded content.
        Use this as the primary search tool for any question — it covers everything:
        definitions, tables, formulas, content passages, and more.
        Works well with both German and English queries.
        Input: a natural language search query or keyword."""
        results = querier.general_search(query)
        if not results:
            return f"No results found for '{query}'."
        return json.dumps(results, ensure_ascii=False, default=str)

    return [
        search,
        lookup_symbols,
        lookup_abbreviations,
        lookup_formula,
        get_unit,
        list_formulas,
    ]
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
