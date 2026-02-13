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
    #  General / broad search
    # ------------------------------------------------------------------ #
    def general_search(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Broad search across all node types. Returns categorised results.
        """
        results: Dict[str, List] = {}

        # Symbols
        syms = self.search_symbols(query)
        if syms:
            results["symbols"] = syms

        # Abbreviations
        abbrs = self.lookup_abbreviation(query)
        if abbrs:
            results["abbreviations"] = abbrs

        # Definitions
        defs = self.search_definitions(query)
        if defs:
            results["definitions"] = defs

        # Units
        units = self.get_unit(query)
        if units:
            results["units"] = units

        # Formulas
        formulas = self.get_formula(query)
        if formulas:
            results["formulas"] = formulas

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
               RETURN docs AS documents, secs AS sections, syms AS symbols,
                      forms AS formulas, abbrs AS abbreviations, units AS units,
                      refs AS references, defs AS definitions""",
        )
        return result[0] if result else {}


# ------------------------------------------------------------------ #
#  Singleton
# ------------------------------------------------------------------ #
_querier: Optional[GraphQuerier] = None


def get_graph_querier() -> GraphQuerier:
    global _querier
    if _querier is None:
        _querier = GraphQuerier()
    return _querier
