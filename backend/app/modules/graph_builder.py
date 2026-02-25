"""Graph Builder – Creates a Neo4j knowledge graph following the Graph-RAG schema.

Schema hierarchy
================
Document → [Volume] → Chapter → Page → Section
  Section → {Table, Figure, Formula, Subsection}
  Section -[:MENTIONS]→ Concept
  Concept ↔ Concept  (RELATED_TO)
  Section ↔ Section  (SEMANTICALLY_SIMILAR)

Every node ultimately belongs to exactly one Document.
Embeddings are stored on Section.embedding and Concept.embedding for hybrid
retrieval (structural + semantic).
"""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from backend.app.modules.database import get_neo4j_connection
from backend.app.modules.latex_utils import latex_to_unicode, normalize_latex
from config.settings import settings

logger = logging.getLogger(__name__)

JSON_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "json"
)

# ===================================================================== #
#  Utility helpers
# ===================================================================== #


def _make_uuid(*parts: str) -> str:
    """Deterministic UUID-5 from concatenated parts."""
    raw = "::".join(str(p) for p in parts)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _figure_fields(fig: dict) -> dict:
    """Derive clean caption, description, and image_path from a raw figure dict.

    The OCR pipeline sometimes stores the image URL in the `description` field
    (when alt text is absent) and leaves `image_url` empty.  This helper
    normalises the data so that:
      • image_path  = actual /api/images/... URL
      • caption     = human-readable text (annotation preferred)
      • description = human-readable text (annotation preferred)
    """
    raw_desc = fig.get("description", "")
    img_url  = fig.get("image_url", "")

    # If description looks like a saved image URL, treat it as the image_path
    if not img_url and raw_desc.startswith("/api/images/"):
        img_url = raw_desc

    annotation = fig.get("annotation", "")

    # Human-readable text: prefer annotation, fall back to description
    # (but never use a URL string as caption/description)
    if raw_desc.startswith("/api/images/"):
        human_text = annotation
    else:
        human_text = raw_desc

    return {
        "caption":     (human_text or annotation)[:200],
        "description": human_text,
        "image_type":  fig.get("type", "image"),
        "annotation":  annotation,
        "image_path":  img_url,
    }


def _detect_document_type(filename: str) -> str:
    lower = filename.lower()
    if "bem" in lower and "ing" in lower:
        return "bem_ing"
    if "handbuch" in lower and "eurocode" in lower:
        return "handbuch_ec"
    if "normen" in lower and "handbuch" in lower:
        return "normen_handbuch"
    if "din" in lower or "en " in lower or "en_" in lower:
        return "din_en"
    return "unknown"


def _detect_eurocode_part(filename: str, content_head: str = "") -> str:
    text = f"{filename} {content_head}".lower()
    for i in range(10):
        for p in (f"ec{i}", f"ec {i}", f"eurocode {i}", f"eurocode{i}", f"en 199{i}"):
            if p in text:
                return f"EC{i}"
    return ""


def _extract_section_number(title: str) -> str:
    """'1.2.3 Definitions' → '1.2.3'  /  'A.1.2 Annex' → 'A.1.2'"""
    m = re.match(r"^([A-Z]?\d+(?:\.\d+)*)\s", title)
    if m:
        return m.group(1)
    m = re.match(r"^([A-Z]\.\d+(?:\.\d+)*)\s", title)
    if m:
        return m.group(1)
    return ""


def _file_checksum(filepath: str) -> str:
    if not filepath or not os.path.exists(filepath):
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ===================================================================== #
#  Domain concept dictionary (German + English civil-engineering terms)
# ===================================================================== #

DOMAIN_CONCEPTS: Dict[str, str] = {
    # German
    "Einwirkung": "Loads and actions applied to structures",
    "Widerstand": "Resistance of structural elements",
    "Bemessung": "Design and dimensioning of structures",
    "Tragfähigkeit": "Load-bearing capacity",
    "Gebrauchstauglichkeit": "Serviceability of structures",
    "Teilsicherheitsbeiwert": "Partial safety factor",
    "Grenzzustand": "Limit state",
    "Tragwerk": "Structure / structural system",
    "Erdbeben": "Earthquake / seismic action",
    "Fundament": "Foundation",
    "Bewehrung": "Reinforcement",
    "Beton": "Concrete",
    "Stahl": "Steel",
    "Spannung": "Stress",
    "Dehnung": "Strain",
    "Verformung": "Deformation",
    "Biegemoment": "Bending moment",
    "Querkraft": "Shear force",
    "Normalkraft": "Normal force",
    "Torsion": "Torsion",
    "Stabilität": "Stability",
    "Knicken": "Buckling",
    "Ermüdung": "Fatigue",
    "Dauerhaftigkeit": "Durability",
    "Brandschutz": "Fire protection",
    "Korrosion": "Corrosion",
    "Setzung": "Settlement",
    "Erdruck": "Earth pressure",
    "Grundbruch": "Bearing capacity failure",
    "Böschungsbruch": "Slope failure",
    "Pfahlgründung": "Pile foundation",
    "Brücke": "Bridge",
    "Tunnel": "Tunnel",
    "Dach": "Roof",
    "Wand": "Wall",
    "Stütze": "Column",
    "Balken": "Beam",
    "Platte": "Slab",
    "Lastfall": "Load case",
    "Lastkombination": "Load combination",
    "Schnittgröße": "Internal force",
    "Sicherheitskonzept": "Safety concept",
    "Zuverlässigkeit": "Reliability",
    "Nachweis": "Verification / proof",
    "Elastizitätsmodul": "Modulus of elasticity",
    "Schubmodul": "Shear modulus",
    "Kriechzahl": "Creep coefficient",
    "Schwinden": "Shrinkage",
    "Vorspannung": "Prestressing",
    "Verbund": "Bond / composite action",
    "Rissbild": "Crack pattern",
    "Rissbreite": "Crack width",
    "Durchstanzen": "Punching shear",
    # English
    "action": "Loads and actions applied to structures",
    "resistance": "Resistance of structural elements",
    "load combination": "Combination of loads for design",
    "partial factor": "Partial safety factor",
    "limit state": "Limit state for design verification",
    "serviceability": "Serviceability limit state (SLS)",
    "ultimate": "Ultimate limit state (ULS)",
    "seismic": "Seismic / earthquake action",
    "foundation": "Foundation / substructure",
    "reinforcement": "Steel reinforcement in concrete",
}


# ===================================================================== #
#  GraphBuilder
# ===================================================================== #


class GraphBuilder:
    """Build a Neo4j knowledge graph following the Graph-RAG schema."""

    def __init__(self):
        self.db = get_neo4j_connection()

    # ----------------------------------------------------------------- #
    #  Public API
    # ----------------------------------------------------------------- #

    def build_all(self) -> Dict[str, Any]:
        """Scan ``json/`` folder, ingest every JSON file into Neo4j."""
        json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))
        if not json_files:
            logger.warning("No JSON files found in %s", JSON_DIR)
            return {"files": 0, "status": "no_files"}

        self._create_constraints_and_indexes()

        total: Dict[str, int] = {
            "files": 0, "documents": 0, "volumes": 0, "chapters": 0,
            "pages": 0, "sections": 0, "tables": 0, "figures": 0,
            "formulas": 0, "concepts": 0,
        }

        for filepath in json_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                stats = self.ingest_document(data, source_path=filepath)
                for k, v in stats.items():
                    total[k] = total.get(k, 0) + v
                total["files"] += 1
            except Exception as e:
                logger.error("Failed to ingest %s: %s", filepath, e)

        logger.info("Graph build complete: %s", total)
        return total

    def clear_graph(self):
        """Remove **all** nodes and relationships."""
        self.db.execute_query("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared")

    # ----------------------------------------------------------------- #
    #  Main ingestion
    # ----------------------------------------------------------------- #

    def ingest_document(
        self,
        data: Dict[str, Any],
        source_path: str = "",
        page_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, int]:
        """Ingest a structured document dict into Neo4j following the schema.

        Handles OCR-produced dicts (chapters, paragraphs, tables, images,
        formulas) and structured JSON files.
        """
        self._create_constraints_and_indexes()

        stats = {
            "documents": 0, "volumes": 0, "chapters": 0, "pages": 0,
            "sections": 0, "tables": 0, "figures": 0, "formulas": 0,
            "concepts": 0,
        }

        # ── Document ────────────────────────────────────────────────────
        doc_name = data.get("document", os.path.basename(source_path))
        filename = data.get("source_file", os.path.basename(source_path))
        language = data.get("language", "de")
        doc_id = _make_uuid("document", doc_name)
        doc_type = _detect_document_type(filename)
        ec_part = _detect_eurocode_part(
            filename,
            (data.get("full_markdown", "") or "")[:3000],
        )
        checksum = _file_checksum(source_path)

        self.db.execute_query(
            """MERGE (d:Document {id: $id})
               SET d.filename         = $filename,
                   d.document_type    = $doc_type,
                   d.eurocode_part    = $ec_part,
                   d.language         = $language,
                   d.upload_timestamp = datetime(),
                   d.version          = $version,
                   d.checksum         = $checksum""",
            {
                "id": doc_id, "filename": filename,
                "doc_type": doc_type, "ec_part": ec_part,
                "language": language,
                "version": data.get("version", "1.0"),
                "checksum": checksum,
            },
        )
        stats["documents"] = 1

        # ── Page nodes (from OCR page_data) ─────────────────────────────
        page_id_map: Dict[int, str] = {}
        if page_data:
            page_id_map = self._create_pages(doc_name, page_data)
            stats["pages"] = len(page_id_map)

        # ── Chapters ────────────────────────────────────────────────────
        raw_chapters = data.get("chapters", [])
        chapter_id_map: Dict[str, str] = {}

        if not raw_chapters:
            # Create a default chapter so sections are always reachable
            raw_chapters = [{
                "title": doc_name,
                "number": "1",
                "section_refs": [
                    s.get("section", "") for s in data.get("sections", [])
                ],
            }]

        for ch in raw_chapters:
            ch_title = ch.get("title", "")
            ch_number = ch.get("number", str(len(chapter_id_map) + 1))
            ch_id = _make_uuid("chapter", doc_name, ch_number)
            chapter_id_map[ch_number] = ch_id

            ch_type = "chapter"
            if re.match(r"^[A-Z]$", ch_number.strip()):
                ch_type = "appendix"
            elif ch_title.lower().startswith(("anhang", "annex", "appendix")):
                ch_type = "appendix"
            elif ch_title.lower().startswith(("vorwort", "foreword", "einleitung")):
                ch_type = "front_matter"

            self.db.execute_query(
                """MERGE (ch:Chapter {id: $id})
                   SET ch.number       = $number,
                       ch.title        = $title,
                       ch.chapter_type = $ch_type,
                       ch.start_page   = $start_page,
                       ch.end_page     = $end_page
                   WITH ch
                   MATCH (d:Document {id: $doc_id})
                   MERGE (d)-[:HAS_CHAPTER]->(ch)""",
                {
                    "id": ch_id, "number": ch_number, "title": ch_title,
                    "ch_type": ch_type,
                    "start_page": ch.get("start_page", 0),
                    "end_page": ch.get("end_page", 0),
                    "doc_id": doc_id,
                },
            )
            stats["chapters"] += 1

        # ── Sections ────────────────────────────────────────────────────
        section_map: Dict[str, Dict] = {}
        prev_by_level: Dict[int, str] = {}
        section_order_on_page: Dict[int, int] = {}

        for sec_data in data.get("sections", []):
            sec_title = sec_data.get("section", "Unknown Section")
            sec_level = sec_data.get("level", 2)
            sec_number = _extract_section_number(sec_title) or sec_title[:30]
            sec_id = _make_uuid("section", doc_name, sec_title)

            # Build full_text from paragraphs + embedded formula LaTeX
            # Formulas are appended verbatim so they are searchable and
            # returned to the LLM as part of the section content.
            paragraphs = sec_data.get("paragraphs", [])
            full_text = " ".join(p.get("text", "") for p in paragraphs).strip()
            if not full_text:
                full_text = sec_data.get("content", "")

            # Append formula expressions so the LLM sees exact LaTeX in context
            formula_parts: List[str] = []
            for frm in sec_data.get("formulas", []):
                expr = frm.get("expression", "") or frm.get("formula", "")
                if expr and expr.strip():
                    formula_parts.append(f"[Formel: {expr.strip()}]")
            if formula_parts:
                full_text = full_text + "\n" + "\n".join(formula_parts)

            content_preview = full_text[:500]

            # Skip heading-only stubs with no meaningful content or child elements
            if (
                len(full_text.strip()) < 50
                and not sec_data.get("tables")
                and not sec_data.get("images")
                and not sec_data.get("formulas")
            ):
                logger.debug("Skipping empty section: '%s'", sec_title)
                continue

            # Page range
            pages_in_sec = sorted(
                {p.get("page", 0) for p in paragraphs if p.get("page")}
            )
            start_page = pages_in_sec[0] if pages_in_sec else 0
            end_page = pages_in_sec[-1] if pages_in_sec else 0

            self.db.execute_query(
                """MERGE (s:Section {id: $id})
                   SET s.number          = $number,
                       s.title           = $title,
                       s.level           = $level,
                       s.start_page      = $start_page,
                       s.end_page        = $end_page,
                       s.content_preview = $preview,
                       s.full_text       = $full_text,
                       s.embedding       = null""",
                {
                    "id": sec_id, "number": sec_number, "title": sec_title,
                    "level": sec_level, "start_page": start_page,
                    "end_page": end_page, "preview": content_preview,
                    "full_text": full_text[:50000],
                },
            )
            stats["sections"] += 1
            section_map[sec_id] = {
                "title": sec_title, "level": sec_level,
                "full_text": full_text, "pages": pages_in_sec,
            }

            # ── Page ↔ Section links ────────────────────────────────────
            if page_id_map and pages_in_sec:
                first_page = pages_in_sec[0]
                for pg in pages_in_sec:
                    if pg not in page_id_map:
                        continue
                    order = section_order_on_page.get(pg, 0)
                    section_order_on_page[pg] = order + 1
                    self.db.execute_query(
                        """MATCH (p:Page {id: $pid})
                           MATCH (s:Section {id: $sid})
                           MERGE (p)-[r:HAS_SECTION]->(s)
                           SET r.order = $order""",
                        {"pid": page_id_map[pg], "sid": sec_id, "order": order},
                    )
                    if pg != first_page:
                        self.db.execute_query(
                            """MATCH (s:Section {id: $sid})
                               MATCH (p:Page {id: $pid})
                               MERGE (s)-[:CONTINUES_ON]->(p)""",
                            {"sid": sec_id, "pid": page_id_map[pg]},
                        )

            # ── Chapter ↔ Section / Page links ───────────────────────────
            for ch in raw_chapters:
                if sec_title in ch.get("section_refs", []):
                    ch_id_val = chapter_id_map.get(ch.get("number", ""))
                    if ch_id_val:
                        # Direct Chapter→Section link (always)
                        self.db.execute_query(
                            """MATCH (ch:Chapter {id: $cid})
                               MATCH (s:Section {id: $sid})
                               MERGE (ch)-[:HAS_SECTION]->(s)""",
                            {"cid": ch_id_val, "sid": sec_id},
                        )
                        # Chapter→Page links (when pages exist)
                        if page_id_map:
                            for pg in pages_in_sec:
                                if pg in page_id_map:
                                    self.db.execute_query(
                                        """MATCH (ch:Chapter {id: $cid})
                                           MATCH (p:Page {id: $pid})
                                           MERGE (ch)-[:CONTAINS_PAGE]->(p)""",
                                        {"cid": ch_id_val, "pid": page_id_map[pg]},
                                    )
                    break

            # ── Subsection hierarchy ────────────────────────────────────
            if sec_level > 1:
                for plevel in range(sec_level - 1, 0, -1):
                    if plevel in prev_by_level:
                        self.db.execute_query(
                            """MATCH (parent:Section {id: $pid})
                               MATCH (child:Section {id: $cid})
                               MERGE (parent)-[:HAS_SUBSECTION]->(child)""",
                            {"pid": prev_by_level[plevel], "cid": sec_id},
                        )
                        break
            prev_by_level[sec_level] = sec_id

            # ── Tables ──────────────────────────────────────────────────
            for i, tbl in enumerate(sec_data.get("tables", [])):
                tbl_id = _make_uuid("table", doc_name, sec_title, str(i))
                self.db.execute_query(
                    """MERGE (t:Table {id: $id})
                       SET t.number        = $number,
                           t.caption       = $caption,
                           t.content       = $content,
                           t.annotation    = $annotation,
                           t.section_title = $section_title,
                           t.embedding     = null
                       WITH t
                       MATCH (s:Section {id: $sid})
                       MERGE (s)-[:HAS_TABLE]->(t)""",
                    {
                        "id": tbl_id,
                        "number": tbl.get("number", str(i + 1)),
                        "caption": tbl.get("caption", ""),
                        "content": (
                            tbl.get("text", "") or tbl.get("html", "")
                        )[:10000],
                        "annotation":    tbl.get("annotation", ""),
                        "section_title": sec_title,
                        "sid": sec_id,
                    },
                )
                stats["tables"] += 1

            # ── Figures ─────────────────────────────────────────────────
            for i, fig in enumerate(sec_data.get("images", [])):
                fig_id = _make_uuid("figure", doc_name, sec_title, str(i))
                self.db.execute_query(
                    """MERGE (f:Figure {id: $id})
                       SET f.number      = $number,
                           f.caption     = $caption,
                           f.description = $description,
                           f.image_type  = $image_type,
                           f.annotation  = $annotation,
                           f.image_path  = $image_path,
                           f.embedding   = null
                       WITH f
                       MATCH (s:Section {id: $sid})
                       MERGE (s)-[:HAS_FIGURE]->(f)""",
                    {
                        "id": fig_id,
                        "number": fig.get("number", str(i + 1)),
                        **_figure_fields(fig),
                        "sid": sec_id,
                    },
                )
                stats["figures"] += 1

            # ── Formulas (section-level) ────────────────────────────────
            for i, frm in enumerate(sec_data.get("formulas", [])):
                frm_id = _make_uuid("formula", doc_name, sec_title, str(i))
                raw_latex = normalize_latex(frm.get("expression", "") or frm.get("formula", ""))
                # Prefer pre-computed unicode from OCR; fallback to converter
                unicode_formula = frm.get("unicode") or latex_to_unicode(raw_latex)
                self.db.execute_query(
                    """MERGE (f:Formula {id: $id})
                       SET f.latex         = $latex,
                           f.unicode       = $unicode,
                           f.section_title = $section_title,
                           f.embedding     = null
                       WITH f
                       MATCH (s:Section {id: $sid})
                       MERGE (s)-[:HAS_FORMULA]->(f)""",
                    {
                        "id": frm_id,
                        "latex": raw_latex,
                        "unicode": unicode_formula,
                        "section_title": sec_title,
                        "sid": sec_id,
                    },
                )
                stats["formulas"] += 1

        # ── Top-level key_formulas (legacy format) ──────────────────────
        for i, kf in enumerate(data.get("key_formulas", [])):
            frm_id = _make_uuid("formula", doc_name, "key", str(i))
            latex = normalize_latex(kf.get("formula", "") or kf.get("expression", ""))
            if not latex:
                continue
            # Prefer pre-computed unicode from OCR; fallback to converter
            unicode_formula = kf.get("unicode") or latex_to_unicode(latex)
            target_sec = None
            ctx = kf.get("name", "")
            for sid, sinfo in section_map.items():
                if ctx and ctx in sinfo.get("title", ""):
                    target_sec = sid
                    break
            if target_sec is None and section_map:
                target_sec = next(iter(section_map))

            key_sec_title = section_map.get(target_sec or "", {}).get("title", "") if section_map else ""
            self.db.execute_query(
                """MERGE (f:Formula {id: $id})
                   SET f.latex         = $latex,
                       f.unicode       = $unicode,
                       f.section_title = $section_title,
                       f.embedding     = null""",
                {"id": frm_id, "latex": latex, "unicode": unicode_formula, "section_title": key_sec_title},
            )
            if target_sec:
                self.db.execute_query(
                    """MATCH (f:Formula {id: $fid})
                       MATCH (s:Section {id: $sid})
                       MERGE (s)-[:HAS_FORMULA]->(f)""",
                    {"fid": frm_id, "sid": target_sec},
                )
            stats["formulas"] += 1

        # ── Concept extraction & MENTIONS edges ─────────────────────────
        concept_count = self._extract_and_create_concepts(
            doc_id, doc_name, data, section_map,
        )
        stats["concepts"] = concept_count

        logger.info("Ingested document '%s': %s", doc_name, stats)
        return stats

    # ----------------------------------------------------------------- #
    #  Page creation
    # ----------------------------------------------------------------- #

    def _create_pages(
        self, doc_name: str, page_data: List[Dict[str, Any]],
    ) -> Dict[int, str]:
        """Create Page nodes and NEXT_PAGE chain.  Return {page_num: page_id}."""
        page_id_map: Dict[int, str] = {}
        for pd in page_data:
            pnum = pd["page_num"]
            pid = _make_uuid("page", doc_name, str(pnum))
            page_id_map[pnum] = pid

            md_lines = pd.get("markdown", "").split("\n")
            header = md_lines[0].strip()[:200] if md_lines else ""
            footer = md_lines[-1].strip()[:200] if len(md_lines) > 1 else ""

            self.db.execute_query(
                """MERGE (p:Page {id: $id})
                   SET p.page_number = $pnum,
                       p.header      = $header,
                       p.footer      = $footer""",
                {"id": pid, "pnum": pnum, "header": header, "footer": footer},
            )

        # NEXT_PAGE chain
        sorted_nums = sorted(page_id_map.keys())
        for i in range(len(sorted_nums) - 1):
            self.db.execute_query(
                """MATCH (p1:Page {id: $id1})
                   MATCH (p2:Page {id: $id2})
                   MERGE (p1)-[:NEXT_PAGE]->(p2)""",
                {
                    "id1": page_id_map[sorted_nums[i]],
                    "id2": page_id_map[sorted_nums[i + 1]],
                },
            )

        return page_id_map

    # ----------------------------------------------------------------- #
    #  Constraints & Indexes
    # ----------------------------------------------------------------- #

    def _create_constraints_and_indexes(self):
        """Create all required constraints, property indexes and full-text indexes."""
        constraints = [
            "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT volume_id_unique IF NOT EXISTS FOR (v:Volume) REQUIRE v.id IS UNIQUE",
            "CREATE CONSTRAINT chapter_id_unique IF NOT EXISTS FOR (ch:Chapter) REQUIRE ch.id IS UNIQUE",
            "CREATE CONSTRAINT page_id_unique IF NOT EXISTS FOR (p:Page) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT section_id_unique IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT table_id_unique IF NOT EXISTS FOR (t:Table) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT figure_id_unique IF NOT EXISTS FOR (f:Figure) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT formula_id_unique IF NOT EXISTS FOR (fm:Formula) REQUIRE fm.id IS UNIQUE",
        ]
        for q in constraints:
            try:
                self.db.execute_query(q)
            except Exception as e:
                logger.debug("Constraint may already exist: %s", e)

        indexes = [
            "CREATE INDEX section_number_idx IF NOT EXISTS FOR (s:Section) ON (s.number)",
            "CREATE INDEX chapter_number_idx IF NOT EXISTS FOR (ch:Chapter) ON (ch.number)",
            "CREATE INDEX concept_norm_idx IF NOT EXISTS FOR (c:Concept) ON (c.normalized_name)",
            "CREATE INDEX page_number_idx IF NOT EXISTS FOR (p:Page) ON (p.page_number)",
            "CREATE INDEX doc_type_idx IF NOT EXISTS FOR (d:Document) ON (d.document_type)",
        ]
        for q in indexes:
            try:
                self.db.execute_query(q)
            except Exception as e:
                logger.debug("Index may already exist: %s", e)

        fulltext_indexes = [
            """CREATE FULLTEXT INDEX section_fulltext IF NOT EXISTS
               FOR (s:Section) ON EACH [s.title, s.full_text, s.content_preview]""",
            """CREATE FULLTEXT INDEX concept_fulltext IF NOT EXISTS
               FOR (c:Concept) ON EACH [c.name, c.description]""",
            """CREATE FULLTEXT INDEX table_fulltext IF NOT EXISTS
               FOR (t:Table) ON EACH [t.caption, t.content, t.section_title]""",
            """CREATE FULLTEXT INDEX formula_fulltext IF NOT EXISTS
               FOR (f:Formula) ON EACH [f.latex, f.unicode, f.section_title]""",
            """CREATE FULLTEXT INDEX figure_fulltext IF NOT EXISTS
               FOR (f:Figure) ON EACH [f.caption, f.description, f.annotation]""",
        ]
        for q in fulltext_indexes:
            try:
                self.db.execute_query(q)
            except Exception:
                pass

    # ----------------------------------------------------------------- #
    #  Concept Extraction
    # ----------------------------------------------------------------- #

    def _extract_and_create_concepts(
        self,
        doc_id: str,
        doc_name: str,
        data: Dict[str, Any],
        section_map: Dict[str, Dict],
    ) -> int:
        """Extract concepts via domain dictionary + legacy fields.

        Creates Concept nodes, MENTIONS edges (Section→Concept) and
        RELATED_TO edges (Concept↔Concept based on co-occurrence).
        """
        count = 0
        created: Dict[str, str] = {}  # normalized_name → concept_id

        def _ensure_concept(name: str, description: str) -> str:
            nonlocal count
            normalized = name.lower().strip()
            if normalized in created:
                return created[normalized]
            cid = _make_uuid("concept", normalized)
            self.db.execute_query(
                """MERGE (c:Concept {id: $id})
                   SET c.name            = $name,
                       c.normalized_name = $normalized,
                       c.description     = $description""",
                {"id": cid, "name": name, "normalized": normalized,
                 "description": description},
            )
            created[normalized] = cid
            count += 1
            return cid

        def _link_mention(sec_id: str, concept_id: str, confidence: float):
            self.db.execute_query(
                """MATCH (s:Section {id: $sid})
                   MATCH (c:Concept {id: $cid})
                   MERGE (s)-[r:MENTIONS]->(c)
                   SET r.confidence = $conf""",
                {"sid": sec_id, "cid": concept_id, "conf": confidence},
            )

        # 1) Domain-dictionary scan over section full_text
        for sec_id, info in section_map.items():
            text = (info.get("full_text", "") or info.get("title", "")).lower()
            for concept_name, desc in DOMAIN_CONCEPTS.items():
                if concept_name.lower() in text:
                    cid = _ensure_concept(concept_name, desc)
                    occ = text.count(concept_name.lower())
                    confidence = min(1.0, 0.3 + occ * 0.1)
                    _link_mention(sec_id, cid, confidence)

        # 2) Legacy fields → Concepts
        for sec_data in data.get("sections", []):
            sec_title = sec_data.get("section", "")
            sec_id = _make_uuid("section", doc_name, sec_title)

            for sym in sec_data.get("symbols", []):
                name = sym.get("symbol", "")
                if name:
                    cid = _ensure_concept(name, sym.get("definition", ""))
                    _link_mention(sec_id, cid, 0.9)

            for defn in sec_data.get("definitions", []):
                term = defn.get("term", "")
                if term:
                    cid = _ensure_concept(term, defn.get("definition", ""))
                    _link_mention(sec_id, cid, 0.95)

            for abbr in sec_data.get("abbreviations", []):
                name = abbr.get("abbreviation", "")
                if name:
                    cid = _ensure_concept(name, abbr.get("definition", ""))
                    _link_mention(sec_id, cid, 0.85)

            for unit in sec_data.get("units", []):
                qty = unit.get("quantity", "")
                if qty:
                    cid = _ensure_concept(qty, f"Unit: {unit.get('unit', '')}")
                    _link_mention(sec_id, cid, 0.8)

        # 3) RELATED_TO between co-occurring concepts
        self._link_related_concepts()

        return count

    def _link_related_concepts(self):
        """Create RELATED_TO edges between Concepts that co-occur in sections."""
        try:
            self.db.execute_query(
                """MATCH (c1:Concept)<-[:MENTIONS]-(s:Section)-[:MENTIONS]->(c2:Concept)
                   WHERE c1.id < c2.id
                   WITH c1, c2, count(s) AS co
                   WHERE co >= 1
                   MERGE (c1)-[r:RELATED_TO]->(c2)
                   SET r.weight = toFloat(co) / 10.0""",
            )
        except Exception as e:
            logger.debug("Could not link related concepts: %s", e)

    # ----------------------------------------------------------------- #
    #  Embedding generation (Section + Concept + Formula)
    # ----------------------------------------------------------------- #

    def generate_embeddings(self, doc_name: Optional[str] = None) -> int:
        """Generate embeddings for Section, Concept, and Formula nodes via Ollama.

        Creates vector indexes if they don't exist.
        Returns total number of embeddings stored.
        """
        from backend.app.modules.ollama_client import get_ollama_client
        ollama = get_ollama_client()
        embedded = 0

        # Ensure vector indexes
        for q in [
            """CREATE VECTOR INDEX section_embedding_index IF NOT EXISTS
               FOR (s:Section) ON (s.embedding)
               OPTIONS {indexConfig: {
                   `vector.dimensions`: 768,
                   `vector.similarity_function`: 'cosine'
               }}""",
            """CREATE VECTOR INDEX concept_embedding_index IF NOT EXISTS
               FOR (c:Concept) ON (c.embedding)
               OPTIONS {indexConfig: {
                   `vector.dimensions`: 768,
                   `vector.similarity_function`: 'cosine'
               }}""",
            """CREATE VECTOR INDEX formula_embedding_index IF NOT EXISTS
               FOR (f:Formula) ON (f.embedding)
               OPTIONS {indexConfig: {
                   `vector.dimensions`: 768,
                   `vector.similarity_function`: 'cosine'
               }}""",
            """CREATE VECTOR INDEX table_embedding_index IF NOT EXISTS
               FOR (t:Table) ON (t.embedding)
               OPTIONS {indexConfig: {
                   `vector.dimensions`: 768,
                   `vector.similarity_function`: 'cosine'
               }}""",
            """CREATE VECTOR INDEX figure_embedding_index IF NOT EXISTS
               FOR (f:Figure) ON (f.embedding)
               OPTIONS {indexConfig: {
                   `vector.dimensions`: 768,
                   `vector.similarity_function`: 'cosine'
               }}""",
        ]:
            try:
                self.db.execute_query(q)
            except Exception as e:
                logger.debug("Vector index note: %s", e)

        # ── Embed Sections ──────────────────────────────────────────────
        sections = self.db.execute_query(
            """MATCH (s:Section)
               WHERE s.embedding IS NULL
               RETURN s.id AS id, s.title AS title, s.full_text AS text
               LIMIT 2000""",
        )

        logger.info("Embedding %d sections …", len(sections))
        for sec in sections:
            title = sec.get("title", "")
            body  = sec.get("text", "") or ""
            # Embed title prominently so short/long sections are found equally well.
            # 8000 chars covers the vast majority of section content without
            # hitting the embedding model's token limit (~8192 tokens).
            text = f"{title}\n\n{body}".strip()
            if not text or len(text.strip()) < 10:
                continue
            try:
                emb = ollama.generate_embedding(text[:8000])
                if emb:
                    self.db.execute_query(
                        """MATCH (s:Section {id: $id}) SET s.embedding = $emb""",
                        {"id": sec["id"], "emb": emb},
                    )
                    embedded += 1
            except Exception as e:
                logger.warning("Section embedding failed %s: %s", sec["id"], e)

        # ── Embed Concepts ──────────────────────────────────────────────
        concepts = self.db.execute_query(
            """MATCH (c:Concept)
               WHERE c.embedding IS NULL
               RETURN c.id AS id, c.name AS name, c.description AS desc
               LIMIT 2000""",
        )

        logger.info("Embedding %d concepts …", len(concepts))
        for con in concepts:
            text = f"{con.get('name', '')} — {con.get('desc', '')}"
            if len(text.strip()) < 5:
                continue
            try:
                emb = ollama.generate_embedding(text)
                if emb:
                    self.db.execute_query(
                        """MATCH (c:Concept {id: $id}) SET c.embedding = $emb""",
                        {"id": con["id"], "emb": emb},
                    )
                    embedded += 1
            except Exception as e:
                logger.warning("Concept embedding failed %s: %s", con["id"], e)

        # ── Embed Formulas ─────────────────────────────────────────────
        formulas = self.db.execute_query(
            """MATCH (s:Section)-[:HAS_FORMULA]->(f:Formula)
               WHERE f.embedding IS NULL
               RETURN f.id AS id,
                      coalesce(f.unicode, f.latex) AS expr,
                      coalesce(f.latex, '')         AS latex,
                      s.title                       AS section_title
               LIMIT 2000""",
        )

        logger.info("Embedding %d formulas …", len(formulas))
        for frm in formulas:
            # Embed section title + expression so the formula is findable by topic.
            # Using unicode where available makes it more semantically readable.
            sec_title = frm.get("section_title", "")
            expr      = frm.get("expr", "") or frm.get("latex", "")
            text = f"{sec_title}\n{expr}".strip() if sec_title else expr
            if len(text.strip()) < 3:
                continue
            try:
                emb = ollama.generate_embedding(text[:1000])
                if emb:
                    self.db.execute_query(
                        """MATCH (f:Formula {id: $id}) SET f.embedding = $emb""",
                        {"id": frm["id"], "emb": emb},
                    )
                    embedded += 1
            except Exception as e:
                logger.warning("Formula embedding failed %s: %s", frm["id"], e)

        # ── Embed Tables ─────────────────────────────────────────────
        tables = self.db.execute_query(
            """MATCH (s:Section)-[:HAS_TABLE]->(t:Table)
               WHERE t.embedding IS NULL
               RETURN t.id AS id,
                      t.caption AS caption,
                      t.content AS content,
                      coalesce(t.annotation, "") AS annotation,
                      s.title AS section_title
               LIMIT 2000""",
        )

        logger.info("Embedding %d tables …", len(tables))
        for tbl in tables:
            text_parts = [
                tbl.get("section_title", ""),
                tbl.get("caption", ""),
                tbl.get("content", ""),
                tbl.get("annotation", ""),
            ]
            text = "\n".join(part for part in text_parts if part)
            if len(text.strip()) < 5:
                continue
            try:
                emb = ollama.generate_embedding(text[:4000])
                if emb:
                    self.db.execute_query(
                        """MATCH (t:Table {id: $id}) SET t.embedding = $emb""",
                        {"id": tbl["id"], "emb": emb},
                    )
                    embedded += 1
            except Exception as e:
                logger.warning("Table embedding failed %s: %s", tbl["id"], e)

        # ── Embed Figures ─────────────────────────────────────────---
        figures = self.db.execute_query(
            """MATCH (s:Section)-[:HAS_FIGURE]->(f:Figure)
               WHERE f.embedding IS NULL
               RETURN f.id AS id,
                      f.caption     AS caption,
                      f.description AS description,
                      coalesce(f.annotation, "") AS annotation,
                      s.title AS section_title
               LIMIT 2000""",
        )

        logger.info("Embedding %d figures …", len(figures))
        for fig in figures:
            text_parts = [
                fig.get("section_title", ""),
                fig.get("caption", ""),
                fig.get("description", ""),
                fig.get("annotation", ""),
            ]
            text = "\n".join(part for part in text_parts if part)
            if len(text.strip()) < 5:
                continue
            try:
                emb = ollama.generate_embedding(text[:1000])
                if emb:
                    self.db.execute_query(
                        """MATCH (f:Figure {id: $id}) SET f.embedding = $emb""",
                        {"id": fig["id"], "emb": emb},
                    )
                    embedded += 1
            except Exception as e:
                logger.warning("Figure embedding failed %s: %s", fig["id"], e)

        logger.info("Stored %d embeddings", embedded)
        return embedded

    # ----------------------------------------------------------------- #
    #  Cross-document semantic similarity
    # ----------------------------------------------------------------- #

    def compute_semantic_similarity(
        self, threshold: float = 0.80, top_k: int = 5,
    ) -> int:
        """Find and store SEMANTICALLY_SIMILAR edges between Sections
        using the vector index."""
        count = 0
        sections = self.db.execute_query(
            """MATCH (s:Section)
               WHERE s.embedding IS NOT NULL
               RETURN s.id AS id, s.embedding AS embedding
               LIMIT 2000""",
        )

        for sec in sections:
            try:
                results = self.db.execute_query(
                    """CALL db.index.vector.queryNodes(
                           'section_embedding_index', $top_k, $embedding
                       ) YIELD node, score
                       WHERE node.id <> $sid AND score >= $threshold
                       WITH node, score
                       MATCH (origin:Section {id: $sid})
                       MERGE (origin)-[r:SEMANTICALLY_SIMILAR]->(node)
                       SET r.score = score
                       RETURN count(r) AS created""",
                    {
                        "top_k": top_k, "embedding": sec["embedding"],
                        "sid": sec["id"], "threshold": threshold,
                    },
                )
                if results:
                    count += results[0].get("created", 0)
            except Exception as e:
                logger.debug("Similarity error for %s: %s", sec["id"], e)
                break  # vector index probably not ready yet

        logger.info("Created %d semantic-similarity links", count)
        return count


# ===================================================================== #
#  Module-level singleton
# ===================================================================== #

_builder: Optional[GraphBuilder] = None


def get_graph_builder() -> GraphBuilder:
    global _builder
    if _builder is None:
        _builder = GraphBuilder()
    return _builder
