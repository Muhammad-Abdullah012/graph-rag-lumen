"""Question Answering Module with Source Attribution"""
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Generator

from backend.app.modules.database import get_neo4j_connection
from backend.app.modules.ollama_client import get_ollama_client
from config.settings import settings

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
CONTEXT_RESERVED_TOKENS = 500  # budget for system prompt + question + answer headroom


def _write_debug_log(system_message: str, prompt: str, answer: str, retrieved_pages: List[Dict[str, Any]]) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_message": system_message,
        "retrieved_pages": retrieved_pages,
        "prompt": prompt,
        "answer": answer,
    }
    try:
        with open(settings.debug_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write debug log: {str(e)}")


SYSTEM_PROMPT = """You are a technical assistant for engineering standards and norms.

STRICT RULES — follow exactly:
1. Answer ONLY using information present in the user-provided context. DO NOT use any general knowledge or outside information.
2. Answer in the same language as the question.
3. Questions may use shorthand (e.g. "λ-Werte") that appears in the context as symbolic notation (e.g. $\\lambda_{\\mathrm{v},1}$) — treat these as the same topic.
4. Copy LaTeX formulas EXACTLY as they appear, preserving all $ and $$ delimiters.
5. Copy pipe-delimited markdown tables EXACTLY, row by row, without modification.
6. State numeric values and norm references exactly as found in the context.
7. If the context does not contain the answer, reply only: "Die angegebenen Dokumente enthalten keine relevanten Informationen zu dieser Frage.\""""

USER_TEMPLATE = """Question: {question}
==========CONTEXT START=============
{context}
===========CONTEXT END==============
"""


class QASystem:
    """Answer questions using knowledge graph and retrieval"""

    def __init__(self):
        self.db = get_neo4j_connection()
        self.ollama = get_ollama_client()

    def answer_question_stream(self, question: str, top_k: int = None) -> Generator[str, None, None]:
        """Stream answer as SSE events: status → token… → done"""
        yield f"data: {json.dumps({'type': 'status', 'message': 'Searching knowledge graph...'})}\n\n"

        question_embedding = self.ollama.generate_embedding(question)
        relevant_pages = self._retrieve_relevant_pages(
            question, question_embedding, top_k or settings.retrieval_top_k
        )

        if not relevant_pages:
            yield f"data: {json.dumps({'type': 'done', 'answer': 'I could not find relevant information to answer this question.', 'sources': [], 'tools_used': []})}\n\n"
            return

        context = self._build_context(relevant_pages)
        sources = self._extract_sources(relevant_pages)

        yield f"data: {json.dumps({'type': 'status', 'message': 'Generating answer...'})}\n\n"

        user_message = USER_TEMPLATE.format(question=question, context=context)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        full_answer = ""
        for token in self.ollama.chat_stream(messages=messages):
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        _write_debug_log(SYSTEM_PROMPT, user_message, full_answer, relevant_pages)
        yield f"data: {json.dumps({'type': 'done', 'answer': full_answer, 'sources': sources, 'tools_used': []})}\n\n"

    def _retrieve_relevant_pages(
        self, question: str, question_embedding: list, top_k: int
    ) -> List[Dict[str, Any]]:
        """Retrieve most similar pages using vector index"""
        query = """
            CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
            YIELD node, score
            MATCH (node)-[:BELONGS_TO]->(doc:Document)
            RETURN
                node.page_number  AS page_number,
                node.markdown     AS markdown,
                node.tables_json  AS tables_json,
                node.header       AS header,
                doc.id            AS document_id,
                doc.name          AS document_name,
                score
            ORDER BY score DESC
        """
        try:
            results = self.db.execute_query(query, {
                "index_name": settings.vector_index_name,
                "top_k": top_k,
                "embedding": question_embedding,
            })
            logger.info(f"Retrieved {len(results)} relevant pages")
            return results
        except Exception as e:
            logger.warning(f"Vector search failed: {str(e)}. Falling back to keyword search.")
            return self._retrieve_pages_by_keyword(question, top_k)

    def _retrieve_pages_by_keyword(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback: retrieve pages using fulltext index"""
        cypher = """
            CALL db.index.fulltext.queryNodes($index_name, $query)
            YIELD node AS p, score
            MATCH (p)-[:BELONGS_TO]->(doc:Document)
            RETURN
                p.page_number  AS page_number,
                p.markdown     AS markdown,
                p.tables_json  AS tables_json,
                p.header       AS header,
                doc.id         AS document_id,
                doc.name       AS document_name,
                score
            LIMIT $top_k
        """
        try:
            return self.db.execute_query(cypher, {
                "index_name": settings.fulltext_index_name,
                "query": query_text,
                "top_k": top_k,
            })
        except Exception as e:
            logger.warning(f"Keyword search failed: {str(e)}")
            return []

    def _build_context(self, pages: List[Dict[str, Any]]) -> str:
        budget_chars = (settings.ollama_num_ctx - CONTEXT_RESERVED_TOKENS) * CHARS_PER_TOKEN

        parts = []
        used = 0
        for page in pages:
            doc_name = page.get("document_name", "Unknown")
            page_num = (page.get("page_number") or 0) + 1
            text = page.get("markdown", "")
            text = self._inline_tables(text, page.get("tables_json") or "[]")
            entry = f"[From {doc_name}, page {page_num}]\n{text}"
            if used + len(entry) > budget_chars:
                break
            parts.append(entry)
            used += len(entry)
        return "\n\n---\n\n".join(parts)

    def _inline_tables(self, markdown: str, tables_json: str) -> str:
        """Replace table placeholders with actual markdown table content."""
        try:
            tables = json.loads(tables_json) if tables_json else []
        except Exception:
            return markdown
        for table in tables:
            table_id = table.get("id", "")
            content = table.get("content", "")
            if table_id and content:
                markdown = markdown.replace(f"[{table_id}]({table_id})", content)
        return markdown

    def _extract_sources(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        seen = set()
        for page in pages:
            doc_id = page.get("document_id")
            page_num = page.get("page_number", 0)
            key = (doc_id, page_num)
            if key not in seen:
                markdown = page.get("markdown", "")
                header = page.get("header") or None
                sources.append({
                    "document": page.get("document_name", "Unknown"),
                    "page_number": (page_num or 0) + 1,
                    "chapter": header,
                    "preview": markdown[:200] if markdown else None,
                })
                seen.add(key)
        return sources
