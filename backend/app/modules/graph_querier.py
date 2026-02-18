"""Graph Querier - Query the Neo4j Eurocode knowledge graph"""
import json
import logging
from typing import List, Dict, Any, Optional

from backend.app.modules.database import get_neo4j_connection

logger = logging.getLogger(__name__)


class GraphQuerier:
    """Query the Eurocode knowledge graph stored in Neo4j."""

    def __init__(self):
        self.db = get_neo4j_connection()

    # ------------------------------------------------------------------ #
    #  Symbol queries
    # ------------------------------------------------------------------ #
    def lookup_symbol(self, symbol_name: str) -> List[Dict[str, Any]]:
        """
        Look up a specific symbol by its exact name (e.g. 'γf', 'Ed', 'Fd').
        Returns all matching symbols with their definition, section and document.
        """
        results = self.db.execute_query(
            """MATCH (sym:Symbol)-[:DEFINED_IN]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
               WHERE sym.name = $name
               OPTIONAL MATCH (sym)-[:RELATED_TO]->(related:Symbol)
               RETURN sym.name AS symbol,
                      sym.definition AS definition,
                      sym.formula AS formula,
                      sym.reference AS reference,
                      sym.anmerkung AS note,
                      sec.name AS section,
                      doc.name AS document,
                      collect(DISTINCT related.name) AS related_symbols""",
            {"name": symbol_name},
        )
        return results

    def search_symbols(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Search symbols whose name or definition contains the given keyword.
        Uses full-text index when available, falls back to CONTAINS.
        """
        # Try full-text first
        try:
            results = self.db.execute_query(
                """CALL db.index.fulltext.queryNodes('symbol_search', $query)
                   YIELD node, score
                   MATCH (node)-[:DEFINED_IN]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
                   RETURN node.name AS symbol,
                          node.definition AS definition,
                          node.formula AS formula,
                          sec.name AS section,
                          doc.name AS document,
                          score
                   ORDER BY score DESC
                   LIMIT 15""",
                {"query": keyword},
            )
            if results:
                return results
        except Exception:
            pass

        # Fallback: case-insensitive CONTAINS
        return self.db.execute_query(
            """MATCH (sym:Symbol)-[:DEFINED_IN]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
               WHERE toLower(sym.name) CONTAINS toLower($kw)
                  OR toLower(sym.definition) CONTAINS toLower($kw)
               RETURN sym.name AS symbol,
                      sym.definition AS definition,
                      sym.formula AS formula,
                      sec.name AS section,
                      doc.name AS document
               LIMIT 15""",
            {"kw": keyword},
        )

    def get_symbols_in_section(self, section_keyword: str) -> List[Dict[str, Any]]:
        """Get all symbols belonging to a specific section (matched by keyword)."""
        return self.db.execute_query(
            """MATCH (sym:Symbol)-[:DEFINED_IN]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
               WHERE toLower(sec.name) CONTAINS toLower($kw)
               RETURN sym.name AS symbol,
                      sym.definition AS definition,
                      sym.formula AS formula,
                      sec.name AS section,
                      doc.name AS document
               ORDER BY sym.name""",
            {"kw": section_keyword},
        )

    # ------------------------------------------------------------------ #
    #  Formula queries
    # ------------------------------------------------------------------ #
    def get_formula(self, formula_name: str) -> List[Dict[str, Any]]:
        """Get a formula by name, including its variables and linked symbols."""
        return self.db.execute_query(
            """MATCH (f:Formula)-[:FROM_DOCUMENT]->(doc:Document)
               WHERE toLower(f.name) CONTAINS toLower($name)
               OPTIONAL MATCH (f)-[:USES_SYMBOL]->(sym:Symbol)
               RETURN f.name AS name,
                      f.expression AS expression,
                      f.variables AS variables,
                      doc.name AS document,
                      collect({symbol: sym.name, definition: sym.definition}) AS used_symbols""",
            {"name": formula_name},
        )

    def list_formulas(self) -> List[Dict[str, Any]]:
        """List all known formulas."""
        return self.db.execute_query(
            """MATCH (f:Formula)-[:FROM_DOCUMENT]->(doc:Document)
               RETURN f.name AS name,
                      f.expression AS expression,
                      doc.name AS document
               ORDER BY f.name""",
        )

    # ------------------------------------------------------------------ #
    #  Abbreviation queries
    # ------------------------------------------------------------------ #
    def lookup_abbreviation(self, abbr: str) -> List[Dict[str, Any]]:
        """Look up an abbreviation (e.g. 'EQU', 'SLS')."""
        results = self.db.execute_query(
            """MATCH (a:Abbreviation)-[:DEFINED_IN]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
               WHERE a.name = $name
               RETURN a.name AS abbreviation,
                      a.definition AS definition,
                      sec.name AS section,
                      doc.name AS document""",
            {"name": abbr},
        )
        if not results:
            # Fallback: case-insensitive search
            results = self.db.execute_query(
                """MATCH (a:Abbreviation)-[:DEFINED_IN]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
                   WHERE toLower(a.name) CONTAINS toLower($kw)
                      OR toLower(a.definition) CONTAINS toLower($kw)
                   RETURN a.name AS abbreviation,
                          a.definition AS definition,
                          sec.name AS section,
                          doc.name AS document
                   LIMIT 10""",
                {"kw": abbr},
            )
        return results

    # ------------------------------------------------------------------ #
    #  Definition queries
    # ------------------------------------------------------------------ #
    def search_definitions(self, keyword: str) -> List[Dict[str, Any]]:
        """Search calculation‐method definitions by keyword."""
        try:
            results = self.db.execute_query(
                """CALL db.index.fulltext.queryNodes('definition_search', $query)
                   YIELD node, score
                   MATCH (node)-[:DEFINED_IN]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
                   RETURN node.term AS term,
                          node.definition AS definition,
                          node.reference AS reference,
                          node.anmerkung AS note,
                          sec.name AS section,
                          doc.name AS document,
                          score
                   ORDER BY score DESC
                   LIMIT 10""",
                {"query": keyword},
            )
            if results:
                return results
        except Exception:
            pass

        return self.db.execute_query(
            """MATCH (df:Definition)-[:DEFINED_IN]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
               WHERE toLower(df.term) CONTAINS toLower($kw)
                  OR toLower(df.definition) CONTAINS toLower($kw)
               RETURN df.term AS term,
                      df.definition AS definition,
                      df.reference AS reference,
                      df.anmerkung AS note,
                      sec.name AS section,
                      doc.name AS document
               LIMIT 10""",
            {"kw": keyword},
        )

    # ------------------------------------------------------------------ #
    #  Unit queries
    # ------------------------------------------------------------------ #
    def get_unit(self, quantity_keyword: str) -> List[Dict[str, Any]]:
        """Get recommended unit for a physical quantity."""
        return self.db.execute_query(
            """MATCH (u:Unit)-[:DEFINED_IN]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
               WHERE toLower(u.quantity) CONTAINS toLower($kw)
               RETURN u.quantity AS quantity,
                      u.unit AS unit,
                      sec.name AS section,
                      doc.name AS document""",
            {"kw": quantity_keyword},
        )

    # ------------------------------------------------------------------ #
    #  Section / Document queries
    # ------------------------------------------------------------------ #
    def list_sections(self, document_keyword: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all sections, optionally filtered by document name keyword."""
        if document_keyword:
            return self.db.execute_query(
                """MATCH (s:Section)-[:BELONGS_TO]->(d:Document)
                   WHERE toLower(d.name) CONTAINS toLower($kw)
                   RETURN s.name AS section, d.name AS document
                   ORDER BY s.name""",
                {"kw": document_keyword},
            )
        return self.db.execute_query(
            """MATCH (s:Section)-[:BELONGS_TO]->(d:Document)
               RETURN s.name AS section, d.name AS document
               ORDER BY d.name, s.name""",
        )

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the graph."""
        return self.db.execute_query(
            """MATCH (d:Document)
               OPTIONAL MATCH (d)<-[:BELONGS_TO]-(s:Section)
               RETURN d.name AS document,
                      d.standard AS standard,
                      count(s) AS section_count
               ORDER BY d.name""",
        )

    # ------------------------------------------------------------------ #
    #  General / broad search  (ALL node types)
    # ------------------------------------------------------------------ #
    def general_search(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Broad search across every node type in the graph.
        Includes: symbols, abbreviations, definitions, units, formulas,
        paragraphs, tables, images, and content blocks (vector + full-text).
        Returns a dict keyed by category.
        """
        results: Dict[str, List] = {}

        # ── structured knowledge nodes ──────────────────────────────────
        syms = self.search_symbols(query)
        if syms:
            results["symbols"] = syms

        abbrs = self.lookup_abbreviation(query)
        if abbrs:
            results["abbreviations"] = abbrs

        defs = self.search_definitions(query)
        if defs:
            results["definitions"] = defs

        units = self.get_unit(query)
        if units:
            results["units"] = units

        formulas = self.get_formula(query)
        if formulas:
            results["formulas"] = formulas

        # ── OCR-extracted content nodes ──────────────────────────────────
        paras = self.search_paragraphs(query, limit=8)
        if paras:
            results["paragraphs"] = paras

        tables = self.search_tables(query, limit=5)
        if tables:
            results["tables"] = tables

        images = self.search_images(query, limit=5)
        if images:
            results["images"] = images

        # ── vector / full-text content blocks ────────────────────────────
        content = self.search_content(query, limit=8)
        if content:
            results["content"] = content

        # ── semantic (vector) search ─────────────────────────────────────
        try:
            semantic = self.semantic_search(query, top_k=6)
            if semantic:
                # Deduplicate against what search_content already returned
                existing_texts = {
                    r.get("text", "")[:80]
                    for r in results.get("content", [])
                }
                unique_semantic = [
                    r for r in semantic
                    if r.get("text", "")[:80] not in existing_texts
                ]
                if unique_semantic:
                    results["semantic"] = unique_semantic
        except Exception as e:
            logger.debug("Semantic search skipped in general_search: %s", e)

        # ── sections & chapters ──────────────────────────────────────────
        sections = self.list_sections(query)
        if sections:
            results["sections"] = sections[:10]

        chapters = self.list_chapters(query)
        if chapters:
            results["chapters"] = chapters[:10]

        return results

    # ------------------------------------------------------------------ #
    #  Graph statistics
    # ------------------------------------------------------------------ #
    def get_graph_stats(self) -> Dict[str, Any]:
        """Return counts of each node type in the graph."""
        result = self.db.execute_query(
            """MATCH (d:Document) WITH count(d) AS docs
               MATCH (s:Section) WITH docs, count(s) AS secs
               MATCH (sym:Symbol) WITH docs, secs, count(sym) AS syms
               OPTIONAL MATCH (f:Formula) WITH docs, secs, syms, count(f) AS forms
               OPTIONAL MATCH (a:Abbreviation) WITH docs, secs, syms, forms, count(a) AS abbrs
               OPTIONAL MATCH (u:Unit) WITH docs, secs, syms, forms, abbrs, count(u) AS units
               OPTIONAL MATCH (r:Reference) WITH docs, secs, syms, forms, abbrs, units, count(r) AS refs
               OPTIONAL MATCH (df:Definition) WITH docs, secs, syms, forms, abbrs, units, refs, count(df) AS defs
               OPTIONAL MATCH (ch:Chapter) WITH docs, secs, syms, forms, abbrs, units, refs, defs, count(ch) AS chapters
               OPTIONAL MATCH (p:Paragraph) WITH docs, secs, syms, forms, abbrs, units, refs, defs, chapters, count(p) AS paragraphs
               OPTIONAL MATCH (t:Table) WITH docs, secs, syms, forms, abbrs, units, refs, defs, chapters, paragraphs, count(t) AS tables
               OPTIONAL MATCH (img:Image) WITH docs, secs, syms, forms, abbrs, units, refs, defs, chapters, paragraphs, tables, count(img) AS images
               OPTIONAL MATCH (cb:ContentBlock) WITH docs, secs, syms, forms, abbrs, units, refs, defs, chapters, paragraphs, tables, images, count(cb) AS content_blocks
               RETURN docs AS documents, secs AS sections, syms AS symbols,
                      forms AS formulas, abbrs AS abbreviations, units AS units,
                      refs AS references, defs AS definitions,
                      chapters AS chapters, paragraphs AS paragraphs,
                      tables AS tables, images AS images,
                      content_blocks AS content_blocks""",
        )
        return result[0] if result else {}

    # ------------------------------------------------------------------ #
    #  Semantic search (vector similarity on ContentBlock embeddings)
    # ------------------------------------------------------------------ #
    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity on ContentBlock embeddings.
        Falls back to full-text search if vector index is not available.
        """
        try:
            from backend.app.modules.ollama_client import get_ollama_client
            ollama = get_ollama_client()
            embedding = ollama.generate_embedding(query)

            if embedding:
                results = self.db.execute_query(
                    """CALL db.index.vector.queryNodes(
                           'content_embedding_index', $top_k, $embedding
                       ) YIELD node, score
                       MATCH (node)-[:FROM_DOCUMENT]->(doc:Document)
                       RETURN node.text AS text,
                              node.section AS section,
                              node.type AS content_type,
                              node.page AS page,
                              doc.name AS document,
                              score
                       ORDER BY score DESC""",
                    {"top_k": top_k, "embedding": embedding},
                )
                if results:
                    return results
        except Exception as e:
            logger.warning("Vector search failed, falling back to text search: %s", e)

        # Fallback: full-text search on paragraphs and content blocks
        return self.search_content(query, top_k)

    def search_content(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Full-text search across Paragraphs and ContentBlocks."""
        results = []

        # Search paragraphs via full-text index
        try:
            para_results = self.db.execute_query(
                """CALL db.index.fulltext.queryNodes('paragraph_search', $query)
                   YIELD node, score
                   MATCH (node)-[:IN_SECTION]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
                   RETURN node.text AS text,
                          sec.name AS section,
                          'paragraph' AS content_type,
                          node.page AS page,
                          doc.name AS document,
                          score
                   ORDER BY score DESC
                   LIMIT $limit""",
                {"query": keyword, "limit": limit},
            )
            results.extend(para_results)
        except Exception:
            pass

        # Fallback: CONTAINS on paragraphs
        if not results:
            try:
                results = self.db.execute_query(
                    """MATCH (p:Paragraph)-[:IN_SECTION]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
                       WHERE toLower(p.text) CONTAINS toLower($kw)
                       RETURN p.text AS text,
                              sec.name AS section,
                              'paragraph' AS content_type,
                              p.page AS page,
                              doc.name AS document,
                              0.5 AS score
                       LIMIT $limit""",
                    {"kw": keyword, "limit": limit},
                )
            except Exception:
                pass

        # Also search content blocks
        try:
            cb_results = self.db.execute_query(
                """CALL db.index.fulltext.queryNodes('content_search', $query)
                   YIELD node, score
                   MATCH (node)-[:FROM_DOCUMENT]->(doc:Document)
                   RETURN node.text AS text,
                          node.section AS section,
                          node.type AS content_type,
                          node.page AS page,
                          doc.name AS document,
                          score
                   ORDER BY score DESC
                   LIMIT $limit""",
                {"query": keyword, "limit": limit},
            )
            results.extend(cb_results)
        except Exception:
            pass

        # De-duplicate and sort by score
        seen = set()
        unique = []
        for r in sorted(results, key=lambda x: x.get("score", 0), reverse=True):
            key = (r.get("text", "")[:100], r.get("document", ""))
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique[:limit]

    # ------------------------------------------------------------------ #
    #  Chapter queries
    # ------------------------------------------------------------------ #
    def list_chapters(self, document_keyword: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all chapters, optionally filtered by document."""
        if document_keyword:
            return self.db.execute_query(
                """MATCH (ch:Chapter)-[:BELONGS_TO]->(d:Document)
                   WHERE toLower(d.name) CONTAINS toLower($kw)
                   RETURN ch.title AS title, ch.number AS number, d.name AS document
                   ORDER BY ch.number""",
                {"kw": document_keyword},
            )
        return self.db.execute_query(
            """MATCH (ch:Chapter)-[:BELONGS_TO]->(d:Document)
               RETURN ch.title AS title, ch.number AS number, d.name AS document
               ORDER BY d.name, ch.number""",
        )

    # ------------------------------------------------------------------ #
    #  Paragraph queries
    # ------------------------------------------------------------------ #
    def search_paragraphs(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search paragraphs by keyword."""
        try:
            results = self.db.execute_query(
                """CALL db.index.fulltext.queryNodes('paragraph_search', $query)
                   YIELD node, score
                   MATCH (node)-[:IN_SECTION]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
                   RETURN node.text AS text,
                          node.page AS page,
                          sec.name AS section,
                          doc.name AS document,
                          score
                   ORDER BY score DESC
                   LIMIT $limit""",
                {"query": keyword, "limit": limit},
            )
            if results:
                return results
        except Exception:
            pass

        return self.db.execute_query(
            """MATCH (p:Paragraph)-[:IN_SECTION]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
               WHERE toLower(p.text) CONTAINS toLower($kw)
               RETURN p.text AS text,
                      p.page AS page,
                      sec.name AS section,
                      doc.name AS document
               LIMIT $limit""",
            {"kw": keyword, "limit": limit},
        )

    # ------------------------------------------------------------------ #
    #  Table queries
    # ------------------------------------------------------------------ #
    def search_tables(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search tables by caption or content keyword."""
        return self.db.execute_query(
            """MATCH (t:Table)-[:IN_SECTION]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
               WHERE toLower(t.caption) CONTAINS toLower($kw)
                  OR toLower(t.html) CONTAINS toLower($kw)
               RETURN t.caption AS caption,
                      t.html AS html,
                      t.page AS page,
                      sec.name AS section,
                      doc.name AS document
               LIMIT $limit""",
            {"kw": keyword, "limit": limit},
        )

    # ------------------------------------------------------------------ #
    #  Image queries
    # ------------------------------------------------------------------ #
    def search_images(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search images by description keyword."""
        return self.db.execute_query(
            """MATCH (img:Image)-[:IN_SECTION]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
               WHERE toLower(img.description) CONTAINS toLower($kw)
               RETURN img.description AS description,
                      img.type AS image_type,
                      img.page AS page,
                      sec.name AS section,
                      doc.name AS document
               LIMIT $limit""",
            {"kw": keyword, "limit": limit},
        )


# ------------------------------------------------------------------ #
#  Singleton
# ------------------------------------------------------------------ #
_querier: Optional[GraphQuerier] = None


def get_graph_querier() -> GraphQuerier:
    global _querier
    if _querier is None:
        _querier = GraphQuerier()
    return _querier
