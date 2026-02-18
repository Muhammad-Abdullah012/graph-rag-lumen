"""Graph Querier – Query the Neo4j Graph-RAG knowledge graph.

Supports
========
- Structural navigation  (Document → Chapter → Page → Section)
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
               OPTIONAL MATCH (d)-[:HAS_CHAPTER]->(:Chapter)-[:CONTAINS_PAGE]->
                              (:Page)-[:HAS_SECTION]->(s:Section)
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
            """MATCH (ch:Chapter {id: $id})-[:CONTAINS_PAGE]->(p:Page)
                     -[:HAS_SECTION]->(s:Section)
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
                """MATCH (d:Document)-[:HAS_CHAPTER]->(:Chapter)-[:CONTAINS_PAGE]->
                         (:Page)-[:HAS_SECTION]->(s:Section)
                   WHERE toLower(d.filename) CONTAINS toLower($kw)
                   RETURN DISTINCT s.title AS section, d.filename AS document
                   ORDER BY s.title
                   UNION
                   MATCH (s:Section)
                   WHERE toLower(s.title) CONTAINS toLower($kw)
                   OPTIONAL MATCH (s)<-[:HAS_SECTION]-(:Page)<-[:CONTAINS_PAGE]-
                                  (:Chapter)<-[:HAS_CHAPTER]-(d:Document)
                   RETURN DISTINCT s.title AS section, d.filename AS document
                   ORDER BY s.title
                   LIMIT 50""",
                {"kw": document_keyword},
            )
        return self.db.execute_query(
            """MATCH (s:Section)
               OPTIONAL MATCH (s)<-[:HAS_SECTION]-(:Page)<-[:CONTAINS_PAGE]-
                              (:Chapter)<-[:HAS_CHAPTER]-(d:Document)
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
                      collect(DISTINCT {id: frm.id, latex: frm.latex}) AS formulas,
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
                   OPTIONAL MATCH (node)<-[:HAS_SECTION]-(:Page)<-[:CONTAINS_PAGE]-
                                  (:Chapter)<-[:HAS_CHAPTER]-(d:Document)
                   RETURN node.id AS id, node.title AS title,
                          node.content_preview AS preview,
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
               OPTIONAL MATCH (s)<-[:HAS_SECTION]-(:Page)<-[:CONTAINS_PAGE]-
                              (:Chapter)<-[:HAS_CHAPTER]-(d:Document)
               RETURN s.id AS id, s.title AS title,
                      s.content_preview AS preview,
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
        """Search concepts by keyword (full-text + fallback)."""
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
               OPTIONAL MATCH (s)<-[:HAS_SECTION]-(:Page)<-[:CONTAINS_PAGE]-
                              (:Chapter)<-[:HAS_CHAPTER]-(d:Document)
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
            results = self.db.execute_query(
                """CALL db.index.fulltext.queryNodes('table_fulltext', $query)
                   YIELD node, score
                   MATCH (s:Section)-[:HAS_TABLE]->(node)
                   OPTIONAL MATCH (s)<-[:HAS_SECTION]-(:Page)<-[:CONTAINS_PAGE]-
                                  (:Chapter)<-[:HAS_CHAPTER]-(d:Document)
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
               OPTIONAL MATCH (s)<-[:HAS_SECTION]-(:Page)<-[:CONTAINS_PAGE]-
                              (:Chapter)<-[:HAS_CHAPTER]-(d:Document)
               RETURN t.caption AS caption, t.content AS content,
                      t.number AS number, s.title AS section,
                      d.filename AS document
               LIMIT $limit""",
            {"kw": keyword, "limit": limit},
        )

    def search_figures(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search figures by caption / description."""
        return self.db.execute_query(
            """MATCH (s:Section)-[:HAS_FIGURE]->(f:Figure)
               WHERE toLower(f.caption) CONTAINS toLower($kw)
                  OR toLower(f.description) CONTAINS toLower($kw)
               OPTIONAL MATCH (s)<-[:HAS_SECTION]-(:Page)<-[:CONTAINS_PAGE]-
                              (:Chapter)<-[:HAS_CHAPTER]-(d:Document)
               RETURN f.caption AS caption, f.description AS description,
                      f.number AS number, f.image_type AS image_type,
                      s.title AS section, d.filename AS document
               LIMIT $limit""",
            {"kw": keyword, "limit": limit},
        )

    def search_formulas(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search formulas by LaTeX content."""
        try:
            results = self.db.execute_query(
                """CALL db.index.fulltext.queryNodes('formula_fulltext', $query)
                   YIELD node, score
                   MATCH (s:Section)-[:HAS_FORMULA]->(node)
                   OPTIONAL MATCH (s)<-[:HAS_SECTION]-(:Page)<-[:CONTAINS_PAGE]-
                                  (:Chapter)<-[:HAS_CHAPTER]-(d:Document)
                   RETURN node.latex AS latex, node.id AS id,
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
               OPTIONAL MATCH (s)<-[:HAS_SECTION]-(:Page)<-[:CONTAINS_PAGE]-
                              (:Chapter)<-[:HAS_CHAPTER]-(d:Document)
               RETURN f.latex AS latex, f.id AS id,
                      s.title AS section, d.filename AS document
               LIMIT $limit""",
            {"kw": keyword, "limit": limit},
        )

    def list_formulas(self) -> List[Dict[str, Any]]:
        """List all formulas."""
        return self.db.execute_query(
            """MATCH (s:Section)-[:HAS_FORMULA]->(f:Formula)
               OPTIONAL MATCH (s)<-[:HAS_SECTION]-(:Page)<-[:CONTAINS_PAGE]-
                              (:Chapter)<-[:HAS_CHAPTER]-(d:Document)
               RETURN f.latex AS latex, f.id AS id,
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
                       OPTIONAL MATCH (node)<-[:HAS_SECTION]-(:Page)<-[:CONTAINS_PAGE]-
                                      (:Chapter)<-[:HAS_CHAPTER]-(d:Document)
                       RETURN node.title AS title,
                              node.content_preview AS preview,
                              node.full_text AS text,
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
               OPTIONAL MATCH (similar)<-[:HAS_SECTION]-(:Page)<-[:CONTAINS_PAGE]-
                              (:Chapter)<-[:HAS_CHAPTER]-(d:Document)
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

        # Concept → Section expansion
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

    # ================================================================== #
    #  9. LEGACY COMPAT (symbols, abbreviations, definitions, units)
    #     These now query Concept nodes that were created from those fields
    # ================================================================== #

    def lookup_symbol(self, symbol_name: str) -> List[Dict[str, Any]]:
        """Look up a symbol (now stored as a Concept) by name."""
        return self.db.execute_query(
            """MATCH (c:Concept)
               WHERE c.name = $name OR c.normalized_name = toLower($name)
               OPTIONAL MATCH (c)<-[m:MENTIONS]-(s:Section)
               OPTIONAL MATCH (s)<-[:HAS_SECTION]-(:Page)<-[:CONTAINS_PAGE]-
                              (:Chapter)<-[:HAS_CHAPTER]-(d:Document)
               RETURN c.name AS symbol, c.description AS definition,
                      s.title AS section, d.filename AS document,
                      m.confidence AS confidence
               ORDER BY m.confidence DESC""",
            {"name": symbol_name},
        )

    def search_symbols(self, keyword: str) -> List[Dict[str, Any]]:
        """Search symbols (concepts) by keyword."""
        return self.search_concepts(keyword, limit=15)

    def get_symbols_in_section(self, section_keyword: str) -> List[Dict[str, Any]]:
        """Get all concepts mentioned in sections matching keyword."""
        return self.db.execute_query(
            """MATCH (s:Section)-[m:MENTIONS]->(c:Concept)
               WHERE toLower(s.title) CONTAINS toLower($kw)
               OPTIONAL MATCH (s)<-[:HAS_SECTION]-(:Page)<-[:CONTAINS_PAGE]-
                              (:Chapter)<-[:HAS_CHAPTER]-(d:Document)
               RETURN c.name AS symbol, c.description AS definition,
                      s.title AS section, d.filename AS document,
                      m.confidence AS confidence
               ORDER BY c.name""",
            {"kw": section_keyword},
        )

    def lookup_abbreviation(self, abbr: str) -> List[Dict[str, Any]]:
        """Look up an abbreviation (now a Concept)."""
        results = self.db.execute_query(
            """MATCH (c:Concept)
               WHERE c.name = $name OR c.normalized_name = toLower($name)
               OPTIONAL MATCH (c)<-[:MENTIONS]-(s:Section)
               RETURN c.name AS abbreviation, c.description AS definition,
                      s.title AS section
               LIMIT 10""",
            {"name": abbr},
        )
        if not results:
            results = self.db.execute_query(
                """MATCH (c:Concept)
                   WHERE toLower(c.name) CONTAINS toLower($kw)
                      OR toLower(c.description) CONTAINS toLower($kw)
                   OPTIONAL MATCH (c)<-[:MENTIONS]-(s:Section)
                   RETURN c.name AS abbreviation, c.description AS definition,
                          s.title AS section
                   LIMIT 10""",
                {"kw": abbr},
            )
        return results

    def search_definitions(self, keyword: str) -> List[Dict[str, Any]]:
        """Search definitions (now Concepts with descriptions)."""
        return self.db.execute_query(
            """MATCH (c:Concept)
               WHERE toLower(c.description) CONTAINS toLower($kw)
                  OR toLower(c.name) CONTAINS toLower($kw)
               OPTIONAL MATCH (c)<-[m:MENTIONS]-(s:Section)
               OPTIONAL MATCH (s)<-[:HAS_SECTION]-(:Page)<-[:CONTAINS_PAGE]-
                              (:Chapter)<-[:HAS_CHAPTER]-(d:Document)
               RETURN c.name AS term, c.description AS definition,
                      s.title AS section, d.filename AS document,
                      m.confidence AS confidence
               ORDER BY m.confidence DESC
               LIMIT 15""",
            {"kw": keyword},
        )

    def get_unit(self, quantity_keyword: str) -> List[Dict[str, Any]]:
        """Get unit info (now a Concept whose description starts with 'Unit:')."""
        return self.db.execute_query(
            """MATCH (c:Concept)
               WHERE (toLower(c.name) CONTAINS toLower($kw)
                   OR toLower(c.description) CONTAINS toLower($kw))
                 AND c.description STARTS WITH 'Unit:'
               RETURN c.name AS quantity,
                      substring(c.description, 6) AS unit
               LIMIT 10""",
            {"kw": quantity_keyword},
        )

    def get_formula(self, formula_keyword: str) -> List[Dict[str, Any]]:
        """Get a formula by keyword."""
        return self.search_formulas(formula_keyword, limit=5)

    # ================================================================== #
    #  10. CONTENT SEARCH (backward compat for agent)
    # ================================================================== #

    def search_content(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Full-text search across section content (replaces ContentBlock search)."""
        return self.search_sections(keyword, limit=limit)

    def search_paragraphs(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search section content (paragraphs are now part of Section.full_text)."""
        return self.db.execute_query(
            """MATCH (s:Section)
               WHERE toLower(s.full_text) CONTAINS toLower($kw)
               OPTIONAL MATCH (s)<-[:HAS_SECTION]-(:Page)<-[:CONTAINS_PAGE]-
                              (:Chapter)<-[:HAS_CHAPTER]-(d:Document)
               RETURN s.full_text AS text,
                      s.start_page AS page,
                      s.title AS section,
                      d.filename AS document
               LIMIT $limit""",
            {"kw": keyword, "limit": limit},
        )

    def search_images(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search figures (images) by description."""
        return self.search_figures(keyword, limit=limit)


# ===================================================================== #
#  Singleton
# ===================================================================== #

_querier: Optional[GraphQuerier] = None


def get_graph_querier() -> GraphQuerier:
    global _querier
    if _querier is None:
        _querier = GraphQuerier()
    return _querier
