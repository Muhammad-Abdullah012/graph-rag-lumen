"""Graph Builder - Creates Neo4j knowledge graph from structured JSON files"""
import json
import logging
import os
import glob
from typing import Dict, Any, Optional

from backend.app.modules.database import get_neo4j_connection
from config.settings import settings

logger = logging.getLogger(__name__)

JSON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "json")


class GraphBuilder:
    """Build a comprehensive Neo4j knowledge graph from structured Eurocode JSON files."""

    def __init__(self):
        self.db = get_neo4j_connection()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def build_all(self) -> Dict[str, Any]:
        """
        Scan the json/ folder, parse every JSON file and ingest into Neo4j.
        Returns a summary dict with counts.
        """
        json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))
        if not json_files:
            logger.warning(f"No JSON files found in {JSON_DIR}")
            return {"files": 0, "status": "no_files"}

        self._create_constraints()

        total_stats: Dict[str, int] = {
            "files": 0,
            "documents": 0,
            "sections": 0,
            "symbols": 0,
            "formulas": 0,
            "definitions": 0,
            "abbreviations": 0,
            "units": 0,
            "references": 0,
        }

        for filepath in json_files:
            try:
                stats = self._ingest_file(filepath)
                for k, v in stats.items():
                    total_stats[k] = total_stats.get(k, 0) + v
                total_stats["files"] += 1
            except Exception as e:
                logger.error(f"Failed to ingest {filepath}: {e}")

        logger.info(f"Graph build complete: {total_stats}")
        return total_stats

    def clear_graph(self):
        """Remove all nodes and relationships (use with care)."""
        self.db.execute_query("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared")

    # ------------------------------------------------------------------ #
    #  Constraints & Indexes
    # ------------------------------------------------------------------ #
    def _create_constraints(self):
        constraints = [
            "CREATE CONSTRAINT doc_name_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT section_id_unique IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT symbol_id_unique IF NOT EXISTS FOR (sym:Symbol) REQUIRE sym.id IS UNIQUE",
            "CREATE CONSTRAINT formula_id_unique IF NOT EXISTS FOR (f:Formula) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT abbreviation_id_unique IF NOT EXISTS FOR (a:Abbreviation) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT reference_name_unique IF NOT EXISTS FOR (r:Reference) REQUIRE r.name IS UNIQUE",
        ]
        for c in constraints:
            try:
                self.db.execute_query(c)
            except Exception as e:
                logger.debug(f"Constraint may already exist: {e}")

        # Full-text indexes for search
        for idx_query in [
            """CREATE FULLTEXT INDEX symbol_search IF NOT EXISTS
               FOR (s:Symbol) ON EACH [s.name, s.definition]""",
            """CREATE FULLTEXT INDEX abbreviation_search IF NOT EXISTS
               FOR (a:Abbreviation) ON EACH [a.name, a.definition]""",
            """CREATE FULLTEXT INDEX definition_search IF NOT EXISTS
               FOR (d:Definition) ON EACH [d.term, d.definition]""",
        ]:
            try:
                self.db.execute_query(idx_query)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  File ingestion
    # ------------------------------------------------------------------ #
    def _ingest_file(self, filepath: str) -> Dict[str, int]:
        """Parse a single JSON file and create graph nodes/relationships."""
        logger.info(f"Ingesting {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        stats = {
            "documents": 0, "sections": 0, "symbols": 0,
            "formulas": 0, "definitions": 0, "abbreviations": 0,
            "units": 0, "references": 0,
        }

        doc_name = data.get("document", os.path.basename(filepath))
        standard = data.get("standard", "")
        iso_ref = data.get("iso_reference", "")

        # Create Document node
        self.db.execute_query(
            """MERGE (d:Document {name: $name})
               SET d.standard = $standard,
                   d.iso_reference = $iso_ref,
                   d.source_file = $source_file""",
            {"name": doc_name, "standard": standard,
             "iso_ref": iso_ref, "source_file": os.path.basename(filepath)},
        )
        stats["documents"] = 1

        # Process sections
        for section_data in data.get("sections", []):
            section_name = section_data.get("section", "Unknown Section")
            section_id = f"{doc_name}::{section_name}"

            self.db.execute_query(
                """MERGE (s:Section {id: $id})
                   SET s.name = $name
                   WITH s
                   MATCH (d:Document {name: $doc_name})
                   MERGE (s)-[:BELONGS_TO]->(d)""",
                {"id": section_id, "name": section_name, "doc_name": doc_name},
            )
            stats["sections"] += 1

            for sym in section_data.get("symbols", []):
                stats["symbols"] += self._create_symbol(sym, section_id, doc_name)
            for defn in section_data.get("definitions", []):
                stats["definitions"] += self._create_definition(defn, section_id, doc_name)
            for abbr in section_data.get("abbreviations", []):
                stats["abbreviations"] += self._create_abbreviation(abbr, section_id, doc_name)
            for unit in section_data.get("units", []):
                stats["units"] += self._create_unit(unit, section_id, doc_name)

        # Key formulas (top-level)
        for formula in data.get("key_formulas", []):
            stats["formulas"] += self._create_formula(formula, doc_name)

        # References
        for ref in data.get("references", []):
            stats["references"] += self._create_reference(ref, doc_name)

        logger.info(f"Ingested {filepath}: {stats}")
        return stats

    # ------------------------------------------------------------------ #
    #  Node creation helpers
    # ------------------------------------------------------------------ #
    def _create_symbol(self, sym: Dict, section_id: str, doc_name: str) -> int:
        symbol_name = sym.get("symbol", "")
        definition = sym.get("definition", "")
        formula = sym.get("formula", "")
        reference = sym.get("reference", "")
        anmerkung = sym.get("anmerkung", "")
        symbol_id = f"{doc_name}::{symbol_name}"

        self.db.execute_query(
            """MERGE (sym:Symbol {id: $id})
               SET sym.name = $name,
                   sym.definition = $definition,
                   sym.formula = $formula,
                   sym.reference = $reference,
                   sym.anmerkung = $anmerkung
               WITH sym
               MATCH (s:Section {id: $section_id})
               MERGE (sym)-[:DEFINED_IN]->(s)
               WITH sym
               MATCH (d:Document {name: $doc_name})
               MERGE (sym)-[:FROM_DOCUMENT]->(d)""",
            {"id": symbol_id, "name": symbol_name, "definition": definition,
             "formula": formula, "reference": reference, "anmerkung": anmerkung,
             "section_id": section_id, "doc_name": doc_name},
        )

        # If the symbol has a formula, link to other symbols it references
        if formula:
            self._link_formula_symbols(symbol_id, formula, doc_name)

        return 1

    def _create_definition(self, defn: Dict, section_id: str, doc_name: str) -> int:
        term = defn.get("term", "")
        definition = defn.get("definition", "")
        reference = defn.get("reference", "")
        anmerkung = defn.get("anmerkung", "")
        defn_id = f"{doc_name}::{term}"

        self.db.execute_query(
            """MERGE (df:Definition {id: $id})
               SET df.term = $term,
                   df.definition = $definition,
                   df.reference = $reference,
                   df.anmerkung = $anmerkung
               WITH df
               MATCH (s:Section {id: $section_id})
               MERGE (df)-[:DEFINED_IN]->(s)
               WITH df
               MATCH (d:Document {name: $doc_name})
               MERGE (df)-[:FROM_DOCUMENT]->(d)""",
            {"id": defn_id, "term": term, "definition": definition,
             "reference": reference, "anmerkung": anmerkung,
             "section_id": section_id, "doc_name": doc_name},
        )
        return 1

    def _create_abbreviation(self, abbr: Dict, section_id: str, doc_name: str) -> int:
        name = abbr.get("abbreviation", "")
        definition = abbr.get("definition", "")
        abbr_id = f"{doc_name}::{name}"

        self.db.execute_query(
            """MERGE (a:Abbreviation {id: $id})
               SET a.name = $name,
                   a.definition = $definition
               WITH a
               MATCH (s:Section {id: $section_id})
               MERGE (a)-[:DEFINED_IN]->(s)
               WITH a
               MATCH (d:Document {name: $doc_name})
               MERGE (a)-[:FROM_DOCUMENT]->(d)""",
            {"id": abbr_id, "name": name, "definition": definition,
             "section_id": section_id, "doc_name": doc_name},
        )
        return 1

    def _create_unit(self, unit: Dict, section_id: str, doc_name: str) -> int:
        quantity = unit.get("quantity", "")
        unit_val = unit.get("unit", "")
        unit_id = f"{doc_name}::{quantity}"

        self.db.execute_query(
            """MERGE (u:Unit {id: $id})
               SET u.quantity = $quantity,
                   u.unit = $unit_val
               WITH u
               MATCH (s:Section {id: $section_id})
               MERGE (u)-[:DEFINED_IN]->(s)
               WITH u
               MATCH (d:Document {name: $doc_name})
               MERGE (u)-[:FROM_DOCUMENT]->(d)""",
            {"id": unit_id, "quantity": quantity, "unit_val": unit_val,
             "section_id": section_id, "doc_name": doc_name},
        )
        return 1

    def _create_formula(self, formula: Dict, doc_name: str) -> int:
        name = formula.get("name", "")
        expression = formula.get("formula", "")
        variables = formula.get("variables", {})
        formula_id = f"{doc_name}::formula::{name}"

        self.db.execute_query(
            """MERGE (f:Formula {id: $id})
               SET f.name = $name,
                   f.expression = $expression,
                   f.variables = $variables
               WITH f
               MATCH (d:Document {name: $doc_name})
               MERGE (f)-[:FROM_DOCUMENT]->(d)""",
            {"id": formula_id, "name": name, "expression": expression,
             "variables": json.dumps(variables, ensure_ascii=False),
             "doc_name": doc_name},
        )

        # Link formula to the symbols it uses
        for var_symbol in variables.keys():
            sym_id = f"{doc_name}::{var_symbol}"
            try:
                self.db.execute_query(
                    """MATCH (f:Formula {id: $formula_id})
                       MATCH (sym:Symbol {id: $sym_id})
                       MERGE (f)-[:USES_SYMBOL]->(sym)""",
                    {"formula_id": formula_id, "sym_id": sym_id},
                )
            except Exception:
                pass

        return 1

    def _create_reference(self, ref: str, doc_name: str) -> int:
        self.db.execute_query(
            """MERGE (r:Reference {name: $name})
               WITH r
               MATCH (d:Document {name: $doc_name})
               MERGE (d)-[:REFERENCES]->(r)""",
            {"name": ref, "doc_name": doc_name},
        )
        return 1

    def _link_formula_symbols(self, symbol_id: str, formula_str: str, doc_name: str):
        """Find symbols referenced in a formula expression and create RELATED_TO edges."""
        try:
            result = self.db.execute_query(
                """MATCH (s:Symbol)-[:FROM_DOCUMENT]->(d:Document {name: $doc_name})
                   RETURN s.name AS name, s.id AS id""",
                {"doc_name": doc_name},
            )
            for row in result:
                other_name = row.get("name", "")
                other_id = row.get("id", "")
                if other_id != symbol_id and other_name and other_name in formula_str:
                    self.db.execute_query(
                        """MATCH (s1:Symbol {id: $id1})
                           MATCH (s2:Symbol {id: $id2})
                           MERGE (s1)-[:RELATED_TO]->(s2)""",
                        {"id1": symbol_id, "id2": other_id},
                    )
        except Exception as e:
            logger.debug(f"Could not link formula symbols: {e}")


# ------------------------------------------------------------------ #
#  Module-level convenience
# ------------------------------------------------------------------ #
_builder: Optional[GraphBuilder] = None


def get_graph_builder() -> GraphBuilder:
    global _builder
    if _builder is None:
        _builder = GraphBuilder()
    return _builder
