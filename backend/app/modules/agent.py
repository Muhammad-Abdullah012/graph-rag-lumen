"""Eurocode Agent - LangGraph ReAct agent with graph-query + semantic-search tools
   and PostgreSQL-backed conversation persistence via LangGraph checkpointer."""
import os
import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from backend.app.modules.graph_querier import get_graph_querier, GraphQuerier
from backend.app.modules.embeddings import get_embedding_service, EmbeddingService
from backend.app.modules.database import get_pg_pool
from config.settings import settings

logger = logging.getLogger(__name__)

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgresdb")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_USER = os.getenv("POSTGRES_USER", "lumenit")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lumenit123")
POSTGRES_DB = os.getenv("POSTGRES_DB", "lumenitdb")

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
    "characters as they appear. The graph stores them as Unicode: γ (gamma), φ (phi), etc.\n\n"
    "You also have a *semantic_search* tool that finds conceptually similar content "
    "even when the exact keyword doesn't match.  Prefer it when keyword search yields "
    "no results, or when the user's query is in natural language."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MESSAGE),
    MessagesPlaceholder(variable_name="messages"),
])


# ================================================================== #
#  Tool definitions
# ================================================================== #
def _setup_tools(querier: GraphQuerier, embed_svc: EmbeddingService):
    """Create LangChain tools wrapping graph queries and semantic search."""

    @tool
    def lookup_symbol(symbol_name: str) -> str:
        """Look up the exact definition of a specific Eurocode symbol by its name.
        Use when the user asks 'What is γf?' or 'Define Ed'.
        Input: the symbol name exactly as written (e.g. 'γf', 'Ed', 'Fd')."""
        results = querier.lookup_symbol(symbol_name)
        if not results:
            return f"No symbol found with name '{symbol_name}'."
        return json.dumps(results, ensure_ascii=False, default=str)

    @tool
    def search_symbols(keyword: str) -> str:
        """Search for symbols whose name or definition contains a keyword.
        Use when the user asks about a concept like 'Teilsicherheitsbeiwert', 'Einwirkung'.
        Input: a search keyword."""
        results = querier.search_symbols(keyword)
        if not results:
            return f"No symbols found matching '{keyword}'."
        return json.dumps(results, ensure_ascii=False, default=str)

    @tool
    def get_symbols_in_section(section_keyword: str) -> str:
        """Get all symbols defined in a specific section.
        Input: a keyword matching the section name."""
        results = querier.get_symbols_in_section(section_keyword)
        if not results:
            return f"No symbols found in section matching '{section_keyword}'."
        return json.dumps(results, ensure_ascii=False, default=str)

    @tool
    def get_formula(formula_name: str) -> str:
        """Get a specific formula by name, including its variables.
        Input: keyword matching the formula name."""
        results = querier.get_formula(formula_name)
        if not results:
            return f"No formula found matching '{formula_name}'."
        return json.dumps(results, ensure_ascii=False, default=str)

    @tool
    def list_formulas() -> str:
        """List all formulas stored in the knowledge graph."""
        results = querier.list_formulas()
        if not results:
            return "No formulas found in the knowledge graph."
        return json.dumps(results, ensure_ascii=False, default=str)

    @tool
    def lookup_abbreviation(abbreviation: str) -> str:
        """Look up the meaning of an abbreviation (e.g. 'EQU', 'SLS', 'ULS').
        Input: the abbreviation."""
        results = querier.lookup_abbreviation(abbreviation)
        if not results:
            return f"No abbreviation found matching '{abbreviation}'."
        return json.dumps(results, ensure_ascii=False, default=str)

    @tool
    def search_definitions(keyword: str) -> str:
        """Search calculation method definitions.
        Input: a search keyword."""
        results = querier.search_definitions(keyword)
        if not results:
            return f"No definitions found matching '{keyword}'."
        return json.dumps(results, ensure_ascii=False, default=str)

    @tool
    def get_unit(quantity_keyword: str) -> str:
        """Get the recommended Eurocode unit for a physical quantity.
        Input: quantity keyword (e.g. 'Kraft', 'Moment')."""
        results = querier.get_unit(quantity_keyword)
        if not results:
            return f"No unit found for quantity '{quantity_keyword}'."
        return json.dumps(results, ensure_ascii=False, default=str)

    @tool
    def list_sections(document_keyword: str = "") -> str:
        """List all sections, optionally filtered by document name.
        Input (optional): document keyword."""
        results = querier.list_sections(document_keyword or None)
        if not results:
            return "No sections found."
        return json.dumps(results, ensure_ascii=False, default=str)

    @tool
    def general_search(query: str) -> str:
        """Broad keyword search across symbols, abbreviations, definitions, units and formulas.
        Use as a fallback when you are not sure which specific tool to call.
        Input: a search query."""
        results = querier.general_search(query)
        if not results:
            return f"No results found for '{query}'."
        return json.dumps(results, ensure_ascii=False, default=str)

    @tool
    async def semantic_search(query: str) -> str:
        """Semantic / vector similarity search across the entire knowledge base.
        Use when keyword search yields no results, or when the user's query is
        in natural language and you need conceptually similar content.
        Input: a natural-language search query."""
        results = await embed_svc.search(query, top_k=10)
        if not results:
            return f"No semantically similar content found for '{query}'."
        return json.dumps(results, ensure_ascii=False, default=str)

    return [
        lookup_symbol,
        search_symbols,
        get_symbols_in_section,
        get_formula,
        list_formulas,
        lookup_abbreviation,
        search_definitions,
        get_unit,
        list_sections,
        general_search,
        semantic_search,
    ]


# ================================================================== #
#  Agent class
# ================================================================== #
class EurocodeAgent:
    """
    LangGraph ReAct agent with:
      • graph-query tools (Neo4j)
      • semantic-search tool (pgvector)
      • persistent conversation history (PostgreSQL checkpointer)
    """

    def __init__(self, checkpointer: AsyncPostgresSaver):
        self.querier = get_graph_querier()
        self.embed_svc = get_embedding_service()
        self.tools = _setup_tools(self.querier, self.embed_svc)
        self.checkpointer = checkpointer

        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_llm_model,
            temperature=0.1,
        )

        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=prompt,
            checkpointer=self.checkpointer,
        )

        logger.info(
            "EurocodeAgent initialised  tools=%d  model=%s  checkpointer=postgres",
            len(self.tools),
            settings.ollama_llm_model,
        )

    # ------------------------------------------------------------------ #
    #  Streaming (SSE)
    # ------------------------------------------------------------------ #
    async def astream_answer(self, question: str, thread_id: str):
        """
        Stream the agent's answer as SSE data lines.
        Yields: `data: <text>\n\n` chunks, tool-call events, and a final `[DONE]`.
        """
        config = {"configurable": {"thread_id": thread_id}}
        existing_state = await self.agent.aget_state(config)
        existing_messages = existing_state.values.get("messages", []) if existing_state else []
        agent_input = {"messages": existing_messages + [("human", question)]}
        tool_names_seen: set = set()

        async for event in self.agent.astream_events(
            agent_input, config=config, version="v2",
        ):
            kind = event.get("event", "")

            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        if isinstance(tc, dict) and "name" in tc:
                            name = tc["name"]
                            if name not in tool_names_seen:
                                yield f"data: {json.dumps({'type': 'tool', 'name': name})}\n\n"
                                tool_names_seen.add(name)

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    # ------------------------------------------------------------------ #
    #  Non-streaming
    # ------------------------------------------------------------------ #
    async def aanswer(self, question: str, thread_id: str) -> Dict[str, Any]:
        """Invoke the agent and return the final answer + tool metadata."""
        config = {"configurable": {"thread_id": thread_id}}
        agent_input = {"messages": [("human", question)]}
        result = await self.agent.ainvoke(agent_input, config=config)

        messages = result.get("messages", [])
        answer = ""
        tools_used: List[Dict[str, Any]] = []

        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_used.append({
                        "tool": tc.get("name", "unknown"),
                        "arguments": tc.get("args", {}),
                    })
            if hasattr(msg, "content") and msg.content and msg.type == "ai":
                answer = msg.content

        return {"answer": answer, "tools_used": tools_used}

    # ------------------------------------------------------------------ #
    #  History retrieval (for loading a conversation)
    # ------------------------------------------------------------------ #
    async def get_thread_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        """Return the message history for a given thread from the checkpointer."""
        config = {"configurable": {"thread_id": thread_id}}
        state = await self.agent.aget_state(config)
        if not state or not state.values:
            return []

        messages = state.values.get("messages", [])
        result = []
        for msg in messages:
            entry: Dict[str, Any] = {
                "role": getattr(msg, "type", "unknown"),
                "content": getattr(msg, "content", ""),
            }
            if entry["role"] == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
                entry["tool_calls"] = [
                    {"tool": tc.get("name", ""), "arguments": tc.get("args", {})}
                    for tc in msg.tool_calls
                ]
            # Skip tool-result messages for the frontend
            if entry["role"] == "tool":
                continue
            result.append(entry)
        return result


# ================================================================== #
#  Singleton (async init required)
# ================================================================== #
_agent: Optional[EurocodeAgent] = None

def get_postgres_connection_string():
    """Build PostgreSQL connection string"""
    return f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

async def init_agent() -> EurocodeAgent:
    """Initialise the singleton agent with a PostgreSQL checkpointer."""
    global _agent
    if _agent is not None:
        return _agent

    DB_URI = get_postgres_connection_string()
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as temp_checkpointer:
        await temp_checkpointer.setup()

    pool = get_pg_pool()
    checkpointer = AsyncPostgresSaver(pool)
    logger.info("LangGraph PostgreSQL checkpointer ready")

    _agent = EurocodeAgent(checkpointer)
    return _agent


def get_agent() -> EurocodeAgent:
    """Return the already-initialised agent singleton.  Raises if not yet inited."""
    if _agent is None:
        raise RuntimeError("Agent not initialised – call init_agent() during startup")
    return _agent
