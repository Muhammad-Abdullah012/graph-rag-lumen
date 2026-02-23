"""Graph Querier – Query the Neo4j Graph-RAG knowledge graph.

Supports
========
- Structural navigation  (Document → Chapter → Section)
- Full-text search        (Section, Concept, Table, Formula)
- Semantic (vector) search on Section and Concept embeddings
- Concept-aware retrieval (MENTIONS, RELATED_TO)
- Cross-document linking  (SEMANTICALLY_SIMILAR)
- Graph statistics
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.app.modules.database import get_neo4j_connection

logger = logging.getLogger(__name__)


class GraphQuerier:
    """Query the Graph-RAG knowledge graph stored in Neo4j."""

    def __init__(self):
        self.db = get_neo4j_connection()

    # ================================================================== #
    #  1. STRUCTURAL NAVIGATION
    # ================================================================== #

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents with chapter and section counts."""
        return self.db.execute_query(
            """MATCH (d:Document)
               OPTIONAL MATCH (d)-[:HAS_CHAPTER]->(ch:Chapter)
               WITH d, count(DISTINCT ch) AS ch_count
               OPTIONAL MATCH (d)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s:Section)
               RETURN d.id AS id,
                      d.filename AS filename,
                      d.document_type AS document_type,
                      d.eurocode_part AS eurocode_part,
                      d.language AS language,
                      ch_count AS chapters,
                      count(DISTINCT s) AS sections
               ORDER BY d.filename""",
        )

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Return a single Document by its id."""
        results = self.db.execute_query(
            """MATCH (d:Document {id: $id})
               RETURN d {.*} AS doc""",
            {"id": doc_id},
        )
        return results[0] if results else None

    def list_chapters(self, document_keyword: Optional[str] = None) -> List[Dict[str, Any]]:
        """List chapters, optionally filtered by document keyword."""
        if document_keyword:
            return self.db.execute_query(
                """MATCH (d:Document)-[:HAS_CHAPTER]->(ch:Chapter)
                   WHERE toLower(d.filename) CONTAINS toLower($kw)
                      OR toLower(d.id) CONTAINS toLower($kw)
                   RETURN ch.id AS id, ch.number AS number, ch.title AS title,
                          ch.chapter_type AS chapter_type, d.filename AS document
                   ORDER BY ch.number""",
                {"kw": document_keyword},
            )
        return self.db.execute_query(
            """MATCH (d:Document)-[:HAS_CHAPTER]->(ch:Chapter)
               RETURN ch.id AS id, ch.number AS number, ch.title AS title,
                      ch.chapter_type AS chapter_type, d.filename AS document
               ORDER BY d.filename, ch.number""",
        )

    def get_chapter_sections(self, chapter_id: str) -> List[Dict[str, Any]]:
        """Get all sections under a chapter (via pages)."""
        return self.db.execute_query(
            """MATCH (ch:Chapter {id: $id})-[:HAS_SECTION]->(s:Section)
               RETURN DISTINCT s.id AS id, s.number AS number, s.title AS title,
                      s.level AS level, s.start_page AS start_page,
                      s.end_page AS end_page,
                      s.content_preview AS preview
               ORDER BY s.start_page, s.number""",
            {"id": chapter_id},
        )

    def list_sections(self, document_keyword: Optional[str] = None) -> List[Dict[str, Any]]:
        """List sections, optionally filtered by document keyword."""
        if document_keyword:
            return self.db.execute_query(
                """MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s:Section)
                   WHERE toLower(d.filename) CONTAINS toLower($kw)
                   RETURN DISTINCT s.title AS section, d.filename AS document
                   ORDER BY s.title
                   UNION
                   MATCH (s:Section)
                   WHERE toLower(s.title) CONTAINS toLower($kw)
                   OPTIONAL MATCH (s)<-[:HAS_SECTION]-(:Chapter)<-[:HAS_CHAPTER]-(d:Document)
                   RETURN DISTINCT s.title AS section, d.filename AS document
                   ORDER BY s.title
                   LIMIT 50""",
                {"kw": document_keyword},
            )
        return self.db.execute_query(
            """MATCH (s:Section)
               OPTIONAL MATCH (s)<-[:HAS_SECTION]-(:Chapter)<-[:HAS_CHAPTER]-(d:Document)
               RETURN DISTINCT s.title AS section, d.filename AS document
               ORDER BY d.filename, s.title
               LIMIT 200""",
        )

    def get_section_detail(self, section_id: str) -> Optional[Dict[str, Any]]:
        """Get full section detail including tables, figures, formulas, subsections."""
        results = self.db.execute_query(
            """MATCH (s:Section {id: $id})
               OPTIONAL MATCH (s)-[:HAS_TABLE]->(t:Table)
               OPTIONAL MATCH (s)-[:HAS_FIGURE]->(f:Figure)
               OPTIONAL MATCH (s)-[:HAS_FORMULA]->(frm:Formula)
               OPTIONAL MATCH (s)-[:HAS_SUBSECTION]->(sub:Section)
               OPTIONAL MATCH (s)-[:MENTIONS]->(c:Concept)
               RETURN s.id AS id, s.title AS title, s.number AS number,
                      s.level AS level, s.full_text AS full_text,
                      s.content_preview AS preview,
                      s.start_page AS start_page, s.end_page AS end_page,
                      collect(DISTINCT {id: t.id, number: t.number,
                              caption: t.caption, content: t.content}) AS tables,
                      collect(DISTINCT {id: f.id, number: f.number,
                              caption: f.caption, description: f.description}) AS figures,
                      collect(DISTINCT {id: frm.id, latex: frm.latex,
                              formula: coalesce(frm.unicode, frm.latex)}) AS formulas,
                      collect(DISTINCT {id: sub.id, title: sub.title,
                              number: sub.number}) AS subsections,
                      collect(DISTINCT {name: c.name,
                              description: c.description}) AS concepts""",
            {"id": section_id},
        )
        return results[0] if results else None

    # ================================================================== #
    #  2. SECTION SEARCH (full-text + keyword)
    # ================================================================== #

    def search_sections(self, keyword: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Search sections by title / content via full-text index, fallback CONTAINS."""
        try:
            results = self.db.execute_query(
                """CALL db.index.fulltext.queryNodes('section_fulltext', $query)
                   YIELD node, score
                   OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(node)
                   RETURN node.id AS id, node.title AS title,
                          substring(node.full_text, 0, 1500) AS content,
                          node.start_page AS page,
                          d.filename AS document, score
                   ORDER BY score DESC
                   LIMIT $limit""",
                {"query": keyword, "limit": limit},
            )
            if results:
                return results
        except Exception:
            pass

        return self.db.execute_query(
            """MATCH (s:Section)
               WHERE toLower(s.title) CONTAINS toLower($kw)
                  OR toLower(s.full_text) CONTAINS toLower($kw)
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
               RETURN s.id AS id, s.title AS title,
                      substring(s.full_text, 0, 1500) AS content,
                      s.start_page AS page,
                      d.filename AS document, 0.5 AS score
               LIMIT $limit""",
            {"kw": keyword, "limit": limit},
        )

    # ================================================================== #
    #  3. CONCEPT QUERIES
    # ================================================================== #

    def lookup_concept(self, name: str) -> List[Dict[str, Any]]:
        """Look up a concept by exact normalized name."""
        return self.db.execute_query(
            """MATCH (c:Concept)
               WHERE c.normalized_name = toLower($name) OR c.name = $name
               OPTIONAL MATCH (c)<-[m:MENTIONS]-(s:Section)
               OPTIONAL MATCH (c)-[r:RELATED_TO]-(rel:Concept)
               RETURN c.name AS concept, c.description AS description,
                      collect(DISTINCT {section: s.title,
                              confidence: m.confidence}) AS mentioned_in,
                      collect(DISTINCT rel.name) AS related_concepts""",
            {"name": name},
        )

    def search_concepts(self, keyword: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Search concepts by keyword.

        Strategy order:
          1. Vector search on concept_embedding_index  (semantic)
          2. Full-text search on concept_fulltext index
          3. CONTAINS fallback
        """
        # Strategy 1: vector search
        try:
            from backend.app.modules.ollama_client import get_ollama_client
            embedding = get_ollama_client().generate_embedding(keyword)
            if embedding:
                results = self.db.execute_query(
                    """CALL db.index.vector.queryNodes(
                           'concept_embedding_index', $limit, $embedding
                       ) YIELD node, score
                       RETURN node.name AS concept, node.description AS description,
                              node.normalized_name AS normalized_name, score
                       ORDER BY score DESC""",
                    {"limit": limit, "embedding": embedding},
                )
                if results:
                    return results
        except Exception:
            pass

        # Strategy 2: full-text search
        try:
            results = self.db.execute_query(
                """CALL db.index.fulltext.queryNodes('concept_fulltext', $query)
                   YIELD node, score
                   RETURN node.name AS concept, node.description AS description,
                          node.normalized_name AS normalized_name, score
                   ORDER BY score DESC
                   LIMIT $limit""",
                {"query": keyword, "limit": limit},
            )
            if results:
                return results
        except Exception:
            pass

        # Strategy 3: CONTAINS fallback
        return self.db.execute_query(
            """MATCH (c:Concept)
               WHERE toLower(c.name) CONTAINS toLower($kw)
                  OR toLower(c.description) CONTAINS toLower($kw)
               RETURN c.name AS concept, c.description AS description,
                      c.normalized_name AS normalized_name, 0.5 AS score
               LIMIT $limit""",
            {"kw": keyword, "limit": limit},
        )

    def get_concept_sections(self, concept_name: str) -> List[Dict[str, Any]]:
        """Get all sections that MENTION a given concept."""
        return self.db.execute_query(
            """MATCH (c:Concept)<-[m:MENTIONS]-(s:Section)
               WHERE c.normalized_name = toLower($name) OR c.name = $name
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
               RETURN s.title AS section, s.content_preview AS preview,
                      m.confidence AS confidence, d.filename AS document,
                      s.start_page AS page
               ORDER BY m.confidence DESC""",
            {"name": concept_name},
        )

    def get_related_concepts(self, concept_name: str) -> List[Dict[str, Any]]:
        """Get concepts related to the given concept (co-occurrence)."""
        return self.db.execute_query(
            """MATCH (c:Concept)-[r:RELATED_TO]-(rel:Concept)
               WHERE c.normalized_name = toLower($name) OR c.name = $name
               RETURN rel.name AS concept, rel.description AS description,
                      r.weight AS weight
               ORDER BY r.weight DESC""",
            {"name": concept_name},
        )

    # ================================================================== #
    #  4. TABLE / FIGURE / FORMULA QUERIES
    # ================================================================== #

    def search_tables(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search tables by caption or content."""
        try:
            from backend.app.modules.ollama_client import get_ollama_client

            embedding = get_ollama_client().generate_embedding(keyword)
            if embedding:
                results = self.db.execute_query(
                    """CALL db.index.vector.queryNodes('table_embedding_index', $limit, $embedding)
                       YIELD node, score
                       MATCH (s:Section)-[:HAS_TABLE]->(node)
                       OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
                       RETURN node.caption AS caption, node.content AS content,
                              node.annotation AS annotation, node.number AS number,
                              node.embedding AS embedding,
                              s.title AS section, d.filename AS document, score
                       ORDER BY score DESC
                       LIMIT $limit""",
                    {"embedding": embedding, "limit": limit},
                )
                if results:
                    return results
        except Exception:
            pass

        try:
            results = self.db.execute_query(
                """CALL db.index.fulltext.queryNodes('table_fulltext', $query)
                   YIELD node, score
                   MATCH (s:Section)-[:HAS_TABLE]->(node)
                   OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
                   RETURN node.caption AS caption, node.content AS content,
                          node.number AS number, s.title AS section,
                          d.filename AS document, score
                   ORDER BY score DESC
                   LIMIT $limit""",
                {"query": keyword, "limit": limit},
            )
            if results:
                return results
        except Exception:
            pass

        return self.db.execute_query(
            """MATCH (s:Section)-[:HAS_TABLE]->(t:Table)
               WHERE toLower(t.caption) CONTAINS toLower($kw)
                  OR toLower(t.content) CONTAINS toLower($kw)
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
               RETURN t.caption AS caption, t.content AS content,
                      t.number AS number, s.title AS section,
                      d.filename AS document
               LIMIT $limit""",
            {"kw": keyword, "limit": limit},
        )

    def search_figures(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search figures by caption / description."""
        try:
            from backend.app.modules.ollama_client import get_ollama_client

            embedding = get_ollama_client().generate_embedding(keyword)
            if embedding:
                results = self.db.execute_query(
                    """CALL db.index.vector.queryNodes('figure_embedding_index', $limit, $embedding)
                       YIELD node, score
                       MATCH (s:Section)-[:HAS_FIGURE]->(node)
                       OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
                       RETURN node.caption AS caption, node.description AS description,
                              node.annotation AS annotation, node.number AS number,
                              node.image_type AS image_type, node.embedding AS embedding,
                              s.title AS section, d.filename AS document, score
                       ORDER BY score DESC
                       LIMIT $limit""",
                    {"embedding": embedding, "limit": limit},
                )
                if results:
                    return results
        except Exception:
            pass

        # Strategy 2: full-text index on figure_fulltext (caption + description + annotation)
        try:
            results = self.db.execute_query(
                """CALL db.index.fulltext.queryNodes('figure_fulltext', $query)
                   YIELD node, score
                   MATCH (s:Section)-[:HAS_FIGURE]->(node)
                   OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
                   RETURN node.caption AS caption, node.description AS description,
                          node.annotation AS annotation, node.number AS number,
                          node.image_type AS image_type,
                          s.title AS section, d.filename AS document, score
                   ORDER BY score DESC
                   LIMIT $limit""",
                {"query": keyword, "limit": limit},
            )
            if results:
                return results
        except Exception:
            pass

        # Strategy 3: CONTAINS fallback
        return self.db.execute_query(
            """MATCH (s:Section)-[:HAS_FIGURE]->(f:Figure)
               WHERE toLower(f.caption) CONTAINS toLower($kw)
                  OR toLower(f.description) CONTAINS toLower($kw)
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
               RETURN f.caption AS caption, f.description AS description,
                      f.number AS number, f.image_type AS image_type,
                      s.title AS section, d.filename AS document
               LIMIT $limit""",
            {"kw": keyword, "limit": limit},
        )

    def search_formulas(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search formulas by LaTeX content.

        Strategy order:
          1. Vector search on formula_embedding_index (semantic — handles natural-language queries)
          2. Full-text search on formula_fulltext index  (keyword/LaTeX fragments)
          3. CONTAINS fallback
        """
        # Strategy 1: vector search (handles "formula for load combination" type queries)
        try:
            from backend.app.modules.ollama_client import get_ollama_client
            embedding = get_ollama_client().generate_embedding(keyword)
            if embedding:
                results = self.db.execute_query(
                    """CALL db.index.vector.queryNodes('formula_embedding_index', $limit, $embedding)
                       YIELD node, score
                       MATCH (s:Section)-[:HAS_FORMULA]->(node)
                       OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
                       RETURN node.latex AS latex,
                              coalesce(node.unicode, node.latex) AS formula,
                              node.embedding AS embedding,
                              node.id AS id,
                              s.title AS section, d.filename AS document, score
                       ORDER BY score DESC
                       LIMIT $limit""",
                    {"embedding": embedding, "limit": limit},
                )
                if results:
                    return results
        except Exception:
            pass

        # Strategy 2: full-text search
        try:
            results = self.db.execute_query(
                """CALL db.index.fulltext.queryNodes('formula_fulltext', $query)
                   YIELD node, score
                   MATCH (s:Section)-[:HAS_FORMULA]->(node)
                   OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
                   RETURN node.latex AS latex,
                          coalesce(node.unicode, node.latex) AS formula,
                          node.embedding AS embedding,
                          node.id AS id,
                          s.title AS section, d.filename AS document, score
                   ORDER BY score DESC
                   LIMIT $limit""",
                {"query": keyword, "limit": limit},
            )
            if results:
                return results
        except Exception:
            pass

        return self.db.execute_query(
            """MATCH (s:Section)-[:HAS_FORMULA]->(f:Formula)
               WHERE toLower(f.latex) CONTAINS toLower($kw)
                  OR toLower(f.unicode) CONTAINS toLower($kw)
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
               RETURN f.latex AS latex,
                      coalesce(f.unicode, f.latex) AS formula,
                      f.embedding AS embedding,
                      f.id AS id,
                      s.title AS section, d.filename AS document
               LIMIT $limit""",
            {"kw": keyword, "limit": limit},
        )

    def list_formulas(self) -> List[Dict[str, Any]]:
        """List all formulas."""
        return self.db.execute_query(
            """MATCH (s:Section)-[:HAS_FORMULA]->(f:Formula)
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(s)
               RETURN f.latex AS latex,
                      coalesce(f.unicode, f.latex) AS formula,
                      f.id AS id,
                      s.title AS section, d.filename AS document
               ORDER BY s.title
               LIMIT 100""",
        )

    # ================================================================== #
    #  5. SEMANTIC (VECTOR) SEARCH
    # ================================================================== #

    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Semantic search over Section embeddings using the vector index.
        Falls back to full-text search if vector index unavailable."""
        try:
            from backend.app.modules.ollama_client import get_ollama_client
            embedding = get_ollama_client().generate_embedding(query)

            if embedding:
                results = self.db.execute_query(
                    """CALL db.index.vector.queryNodes(
                           'section_embedding_index', $top_k, $embedding
                       ) YIELD node, score
                       OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(node)
                       RETURN node.title AS title,
                              substring(node.full_text, 0, 1500) AS content,
                              node.start_page AS page,
                              d.filename AS document,
                              score
                       ORDER BY score DESC""",
                    {"top_k": top_k, "embedding": embedding},
                )
                if results:
                    return results
        except Exception as e:
            logger.warning("Vector search failed, falling back: %s", e)

        # Fallback: full-text
        return self.search_sections(query, limit=top_k)

    def semantic_concept_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Semantic search over Concept embeddings."""
        try:
            from backend.app.modules.ollama_client import get_ollama_client
            embedding = get_ollama_client().generate_embedding(query)

            if embedding:
                results = self.db.execute_query(
                    """CALL db.index.vector.queryNodes(
                           'concept_embedding_index', $top_k, $embedding
                       ) YIELD node, score
                       RETURN node.name AS concept,
                              node.description AS description,
                              score
                       ORDER BY score DESC""",
                    {"top_k": top_k, "embedding": embedding},
                )
                if results:
                    return results
        except Exception as e:
            logger.warning("Concept vector search failed: %s", e)

        return self.search_concepts(query, limit=top_k)

    # ================================================================== #
    #  6. CROSS-DOCUMENT QUERIES
    # ================================================================== #

    def get_similar_sections(self, section_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get sections semantically similar to a given section (cross-document)."""
        return self.db.execute_query(
            """MATCH (s:Section {id: $id})-[r:SEMANTICALLY_SIMILAR]-(similar:Section)
               OPTIONAL MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(similar)
               RETURN similar.title AS title,
                      similar.content_preview AS preview,
                      d.filename AS document,
                      r.score AS score,
                      similar.start_page AS page
               ORDER BY r.score DESC
               LIMIT $limit""",
            {"id": section_id, "limit": limit},
        )

    # ================================================================== #
    #  7. GENERAL / BROAD SEARCH (combines everything)
    # ================================================================== #

    def general_search(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """Broad search across every node type in the graph.

        Returns a dict keyed by category: sections, concepts, tables,
        figures, formulas, semantic, similar, chapters.
        """
        results: Dict[str, List] = {}

        # ── Sections (full-text) ────────────────────────────────────────
        secs = self.search_sections(query, limit=10)
        if secs:
            results["sections"] = secs

        # ── Concepts ────────────────────────────────────────────────────
        concepts = self.search_concepts(query, limit=10)
        if concepts:
            results["concepts"] = concepts

        # Concept → Section expansion (full-text concepts)
        for c in (concepts or [])[:3]:
            cname = c.get("concept", "")
            if cname:
                csecs = self.get_concept_sections(cname)
                if csecs:
                    existing = {r.get("section", "") for r in results.get("sections", [])}
                    for cs in csecs:
                        if cs.get("section", "") not in existing:
                            results.setdefault("sections", []).append(cs)

        # ── Tables ──────────────────────────────────────────────────────
        tables = self.search_tables(query, limit=5)
        if tables:
            results["tables"] = tables

        # ── Figures ─────────────────────────────────────────────────────
        figures = self.search_figures(query, limit=5)
        if figures:
            results["figures"] = figures

        # ── Formulas ────────────────────────────────────────────────────
        formulas = self.search_formulas(query, limit=5)
        if formulas:
            results["formulas"] = formulas

        # ── Semantic (vector) search ────────────────────────────────────
        try:
            semantic = self.semantic_search(query, top_k=8)
            if semantic:
                existing_titles = {
                    r.get("title", "") for r in results.get("sections", [])
                }
                unique = [r for r in semantic if r.get("title", "") not in existing_titles]
                if unique:
                    results["semantic"] = unique
        except Exception as e:
            logger.debug("Semantic search skipped in general_search: %s", e)

        # ── Chapters ────────────────────────────────────────────────────
        chapters = self.list_chapters(query)
        if chapters:
            results["chapters"] = chapters[:10]

        return results

    # ================================================================== #
    #  8. GRAPH STATISTICS
    # ================================================================== #

    def get_graph_stats(self) -> Dict[str, Any]:
        """Return counts of each node type plus relationship counts."""
        result = self.db.execute_query(
            """OPTIONAL MATCH (doc:Document) WITH count(doc) AS documents
               OPTIONAL MATCH (v:Volume) WITH documents, count(v) AS volumes
               OPTIONAL MATCH (ch:Chapter) WITH documents, volumes, count(ch) AS chapters
               OPTIONAL MATCH (p:Page) WITH documents, volumes, chapters, count(p) AS pages
               OPTIONAL MATCH (s:Section) WITH documents, volumes, chapters, pages, count(s) AS sections
               OPTIONAL MATCH (t:Table) WITH documents, volumes, chapters, pages, sections, count(t) AS tables
               OPTIONAL MATCH (f:Figure) WITH documents, volumes, chapters, pages, sections, tables, count(f) AS figures
               OPTIONAL MATCH (fm:Formula) WITH documents, volumes, chapters, pages, sections, tables, figures, count(fm) AS formulas
               OPTIONAL MATCH (c:Concept) WITH documents, volumes, chapters, pages, sections, tables, figures, formulas, count(c) AS concepts
               RETURN documents, volumes, chapters, pages, sections,
                      tables, figures, formulas, concepts""",
        )
        stats = result[0] if result else {}

        # Relationship counts
        try:
            rel_result = self.db.execute_query(
                """OPTIONAL MATCH ()-[m:MENTIONS]->() WITH count(m) AS mentions
                   OPTIONAL MATCH ()-[r:RELATED_TO]->() WITH mentions, count(r) AS related_to
                   OPTIONAL MATCH ()-[ss:SEMANTICALLY_SIMILAR]->() WITH mentions, related_to, count(ss) AS semantic_links
                   RETURN mentions, related_to, semantic_links""",
            )
            if rel_result:
                stats.update(rel_result[0])
        except Exception:
            pass

        return stats


# ===================================================================== #
#  Singleton
# ===================================================================== #

_querier: Optional[GraphQuerier] = None


def get_graph_querier() -> GraphQuerier:
    global _querier
    if _querier is None:
        _querier = GraphQuerier()
    return _querier
