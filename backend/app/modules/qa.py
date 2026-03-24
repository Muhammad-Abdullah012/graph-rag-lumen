"""Question Answering Module with Source Attribution"""
import json
import logging
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Generator, Tuple

from backend.app.modules.database import get_neo4j_connection
from backend.app.modules.ollama_client import get_ollama_client
from config.settings import settings

logger = logging.getLogger(__name__)


def _write_debug_log(
    system_message: str,
    prompt: str,
    answer: str,
    retrieved_pages: Dict[str, Any],
    draft_answer: str = "",
    extracted_refs: List[str] = None,
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_message": system_message,
        "vector_hits": retrieved_pages.get("vector_hits", []),
        "fulltext_hits": retrieved_pages.get("fulltext_hits", []),
        "merged_pages": retrieved_pages.get("merged_pages", []),
        "draft_answer": draft_answer,
        "extracted_refs": extracted_refs or [],
        "resolved_references": retrieved_pages.get("resolved_references", []),
        "reference_pages": retrieved_pages.get("reference_pages", []),
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

REFERENCE_EXTRACTION_PROMPT = """You are analyzing a technical draft answer about engineering standards.

Look at the draft answer below. Identify any norm sections or standards that are:
- Mentioned or cited (e.g. "DIN EN 1993-2:2010-12, 9.5.2", "EC 3-2 9.5.2")
- But whose specific content (formulas, values, tables) is NOT included in the answer

Return ONLY a JSON array of SECTION-LEVEL search strings (do NOT include sub-paragraph numbers like (3) or (4)).
Group all sub-paragraphs of the same section into a single entry.
Example: ["DIN EN 1993-2 9.5.2"] — NOT ["DIN EN 1993-2 9.5.2(3)", "DIN EN 1993-2 9.5.2(4)"]
Return [] if nothing is missing or the answer is already complete.

Draft answer:
{draft}"""


FINAL_USER_TEMPLATE = """Question: {question}

A draft answer was produced from initial search results:
--- DRAFT ANSWER START ---
{draft}
--- DRAFT ANSWER END ---

Additional reference pages were retrieved via cross-references. Use ALL relevant information from both the draft and the context below to produce a complete final answer.
Do NOT omit formulas, values, or norm references that appear in the draft.

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
        try:
            yield from self._answer_question_stream_inner(question, top_k)
        except Exception as e:
            logger.error(f"Unhandled error in answer_question_stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    def _answer_question_stream_inner(self, question: str, top_k: int = None) -> Generator[str, None, None]:
        """Inner generator — all exceptions propagate to answer_question_stream."""
        yield f"data: {json.dumps({'type': 'status', 'message': 'Searching knowledge graph...'})}\n\n"

        question_embedding = self.ollama.generate_embedding(question)
        retrieval = self._retrieve_relevant_pages(
            question, question_embedding, top_k or settings.retrieval_top_k
        )
        relevant_pages = retrieval["merged_pages"]

        if not relevant_pages:
            yield f"data: {json.dumps({'type': 'done', 'answer': 'I could not find relevant information to answer this question.', 'sources': [], 'tools_used': []})}\n\n"
            return

        # --- Step 2: First LLM call — draft answer (streamed internally, not shown to user) ---
        # Stream tokens from Ollama and accumulate them as the draft.
        # Emit a keepalive status event every N tokens so proxies/browsers don't
        # close the SSE connection while waiting for the slow LLM.
        yield f"data: {json.dumps({'type': 'status', 'message': 'Generating draft answer...'})}\n\n"
        context = self._build_context(relevant_pages)
        user_message = USER_TEMPLATE.format(question=question, context=context)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        draft = ""
        token_count = 0
        for token in self.ollama.chat_stream(messages, temperature=0.0, think=False):
            draft += token
            token_count += 1
            if token_count % settings.draft_keepalive_interval == 0:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Generating draft answer...'})}\n\n"
        draft = draft.strip()

        # --- Step 3: Extract missing references from draft ---
        extracted_refs = self._extract_references_from_draft(draft)

        ref_pages: List[Dict[str, Any]] = []
        resolved_refs: List[Dict[str, Any]] = []

        if extracted_refs:
            yield f"data: {json.dumps({'type': 'status', 'message': f'Following {len(extracted_refs)} reference(s)...'})}\n\n"
            seen_ids = {p["page_id"] for p in relevant_pages if p.get("page_id")}
            ref_pages, resolved_refs = self._fetch_reference_pages(extracted_refs, seen_ids)
            if ref_pages:
                logger.info(f"Reference expansion: {len(extracted_refs)} ref(s) → {len(ref_pages)} new page(s)")

        retrieval["resolved_references"] = resolved_refs
        retrieval["reference_pages"] = ref_pages

        # --- Step 4/5: Final answer ---
        if ref_pages:
            # Regenerate: pass draft + only reference pages as context
            # (original pages are already summarised in the draft; adding them again would dilute with noise)
            yield f"data: {json.dumps({'type': 'status', 'message': 'Regenerating with additional references...'})}\n\n"
            ref_context = self._build_context(ref_pages)
            final_user_message = FINAL_USER_TEMPLATE.format(
                question=question, draft=draft, context=ref_context
            )
            final_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": final_user_message},
            ]
            full_answer = ""
            for token in self.ollama.chat_stream(messages=final_messages):
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        else:
            # No new references found — stream the draft as the final answer
            full_answer = draft
            yield f"data: {json.dumps({'type': 'token', 'content': draft})}\n\n"

        sources = self._extract_sources(relevant_pages + ref_pages)
        _write_debug_log(SYSTEM_PROMPT, user_message, full_answer, retrieval, draft, extracted_refs)
        yield f"data: {json.dumps({'type': 'done', 'answer': full_answer, 'sources': sources, 'tools_used': []})}\n\n"

    def _retrieve_relevant_pages(
        self, question: str, question_embedding: list, top_k: int
    ) -> Dict[str, Any]:
        """Run vector + fulltext search, merge with RRF, expand with neighbors.
        Returns a dict with all intermediate results for debugging."""
        vector_hits = self._vector_search(question_embedding, top_k)
        fulltext_hits = self._fulltext_search(question, top_k)

        merged = self._rrf_merge(vector_hits, fulltext_hits, top_k)
        logger.info(f"RRF merge: {len(vector_hits)} vector + {len(fulltext_hits)} fulltext → {len(merged)} merged")

        final = self._expand_with_neighbors(merged) if merged and settings.retrieval_neighbor_pages > 0 else merged

        return {
            "vector_hits": vector_hits,
            "fulltext_hits": fulltext_hits,
            "top_k_pages": merged,       # RRF top-K only, before neighbor expansion
            "merged_pages": final,       # top-K + neighbor-expanded pages
        }

    def _vector_search(self, embedding: list, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve pages via vector similarity index."""
        query = """
            CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
            YIELD node, score
            MATCH (node)-[:BELONGS_TO]->(doc:Document)
            RETURN
                node.id           AS page_id,
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
            return self.db.execute_query(query, {
                "index_name": settings.vector_index_name,
                "top_k": top_k,
                "embedding": embedding,
            })
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    def _fulltext_search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve pages via fulltext (BM25) index."""
        cypher = """
            CALL db.index.fulltext.queryNodes($index_name, $query)
            YIELD node AS p, score
            MATCH (p)-[:BELONGS_TO]->(doc:Document)
            RETURN
                p.id           AS page_id,
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
            logger.warning(f"Fulltext search failed: {e}")
            return []

    def _rrf_merge(
        self, vector_hits: List[Dict], fulltext_hits: List[Dict], top_k: int
    ) -> List[Dict[str, Any]]:
        """Merge two ranked lists using Reciprocal Rank Fusion."""
        k = settings.rrf_k
        rrf_scores: Dict[str, float] = {}
        page_data: Dict[str, Dict] = {}

        for rank, hit in enumerate(vector_hits):
            pid = hit.get("page_id")
            if not pid:
                continue
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
            page_data[pid] = hit

        for rank, hit in enumerate(fulltext_hits):
            pid = hit.get("page_id")
            if not pid:
                continue
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
            if pid not in page_data:
                page_data[pid] = hit

        sorted_ids = sorted(rrf_scores, key=lambda p: rrf_scores[p], reverse=True)
        return [page_data[pid] for pid in sorted_ids]

    def _expand_with_neighbors(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Append following pages (via NEXT_PAGE) for the top N hits."""
        depth = settings.retrieval_neighbor_pages
        neighbor_query = f"""
            MATCH (start:Page {{id: $page_id}})-[:NEXT_PAGE*1..{depth}]->(nb:Page)
            MATCH (nb)-[:BELONGS_TO]->(doc:Document)
            RETURN
                nb.id           AS page_id,
                nb.page_number  AS page_number,
                nb.markdown     AS markdown,
                nb.tables_json  AS tables_json,
                nb.header       AS header,
                doc.id          AS document_id,
                doc.name        AS document_name,
                $base_score     AS score
        """
        seen_ids = {h["page_id"] for h in hits}
        extra = []
        for hit in hits[:settings.retrieval_neighbor_expand_top]:
            try:
                neighbors = self.db.execute_query(neighbor_query, {
                    "page_id": hit["page_id"],
                    "base_score": hit["score"],
                })
                for nb in neighbors:
                    if nb["page_id"] not in seen_ids:
                        seen_ids.add(nb["page_id"])
                        extra.append(nb)
            except Exception as e:
                logger.warning(f"Neighbor expansion failed for page {hit.get('page_id')}: {e}")

        results = hits + extra
        logger.info(f"Neighbor expansion: {len(hits)} hits + {len(extra)} neighbors = {len(results)} total")
        return results

    def _extract_references_from_draft(self, draft: str) -> List[str]:
        """Ask the LLM to identify norm sections cited in the draft but not fully answered.

        Returns a list of search strings (e.g. ["DIN EN 1993-2 9.5.2"]) or [] if complete.
        """
        prompt = REFERENCE_EXTRACTION_PROMPT.format(draft=draft)
        try:
            raw = self.ollama.chat([{"role": "user", "content": prompt}], temperature=0.0)
        except Exception as e:
            logger.warning(f"Reference extraction LLM call failed: {e}")
            return []

        # Strip any residual <think>...</think> blocks
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Extract JSON array from response (handles markdown code fences)
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            logger.info("Reference extraction: no JSON array found in response")
            return []

        try:
            refs = json.loads(match.group())
            if isinstance(refs, list):
                result = [str(r).strip() for r in refs if r]
                logger.info(f"Reference extraction: {result}")
                return result[:settings.reference_extraction_max_refs]
        except json.JSONDecodeError:
            logger.warning(f"Reference extraction: failed to parse JSON: {match.group()!r}")

        return []

    def _fetch_reference_pages(
        self, ref_strings: List[str], seen_ids: set
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """For each extracted reference string, search Reference nodes via fulltext,
        follow RESOLVED_TO to target pages, and expand 1 NEXT_PAGE neighbor.

        Returns (new_pages, resolved) where resolved logs which ref → which page.
        """
        ref_cypher = """
            CALL db.index.fulltext.queryNodes($ref_index, $ref_string)
            YIELD node AS r, score
            WHERE score >= $threshold
            MATCH (r)-[:RESOLVED_TO]->(target:Page)
            WHERE NOT target.id IN $seen_ids
            MATCH (target)-[:BELONGS_TO]->(d:Document)
            RETURN DISTINCT
                target.id           AS page_id,
                target.page_number  AS page_number,
                target.markdown     AS markdown,
                target.tables_json  AS tables_json,
                target.header       AS header,
                d.id                AS document_id,
                d.name              AS document_name,
                r.full_reference    AS _source_ref
            LIMIT $max_pages
        """
        neighbor_depth = settings.reference_neighbor_pages
        neighbor_cypher = f"""
            MATCH (start:Page {{id: $page_id}})-[:NEXT_PAGE*1..{neighbor_depth}]->(nb:Page)
            MATCH (nb)-[:BELONGS_TO]->(doc:Document)
            RETURN
                nb.id           AS page_id,
                nb.page_number  AS page_number,
                nb.markdown     AS markdown,
                nb.tables_json  AS tables_json,
                nb.header       AS header,
                doc.id          AS document_id,
                doc.name        AS document_name,
                $base_score     AS score
        """

        local_seen = set(seen_ids)
        all_pages: List[Dict[str, Any]] = []
        all_resolved: List[Dict[str, Any]] = []

        for ref_str in ref_strings:
            try:
                rows = self.db.execute_query(ref_cypher, {
                    "ref_index": settings.reference_fulltext_index_name,
                    "ref_string": ref_str,
                    "threshold": settings.reference_resolve_score_threshold,
                    "seen_ids": list(local_seen),
                    "max_pages": settings.reference_extraction_max_pages,
                })
            except Exception as e:
                logger.warning(f"Reference fetch failed for '{ref_str}': {e}")
                continue

            for row in rows:
                pid = row.get("page_id")
                if not pid or pid in local_seen:
                    continue
                local_seen.add(pid)
                page = {k: v for k, v in row.items() if not k.startswith("_")}
                all_pages.append(page)
                all_resolved.append({
                    "ref_string": ref_str,
                    "full_reference": row.get("_source_ref"),
                    "target_page_id": pid,
                })

                # Expand NEXT_PAGE neighbors of the resolved page
                if neighbor_depth > 0:
                    try:
                        neighbors = self.db.execute_query(neighbor_cypher, {
                            "page_id": pid,
                            "base_score": 0.0,
                        })
                        for nb in neighbors:
                            nb_id = nb.get("page_id")
                            if nb_id and nb_id not in local_seen:
                                local_seen.add(nb_id)
                                all_pages.append(nb)
                    except Exception as e:
                        logger.warning(f"Neighbor expansion failed for ref page {pid}: {e}")

        logger.info(
            f"Reference fetch: {len(ref_strings)} string(s) → "
            f"{len(all_resolved)} resolved ref(s) → {len(all_pages)} page(s)"
        )
        return all_pages, all_resolved

    def _build_context(self, pages: List[Dict[str, Any]]) -> str:
        budget_chars = (settings.ollama_num_ctx - settings.context_reserved_tokens) * settings.chars_per_token

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
