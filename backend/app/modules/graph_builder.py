"""Graph Builder – Creates a Neo4j knowledge graph following the Graph-RAG schema.

Schema hierarchy
================
Document → [Volume] → Chapter → Page → Section
  Section → {Table, Figure, Formula, Subsection}
  Section ↔ Section  (SEMANTICALLY_SIMILAR)

Every node ultimately belongs to exactly one Document.
Document.id  = exact source filename including UUID prefix (stable unique key).
Document.name = human-readable name stripped of UUID prefix and file extension.
Embeddings are stored on Section, Page, Chapter, and Document nodes for hybrid
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


_UUID_PREFIX_RE = re.compile(r'^[a-f0-9]{32}_', re.IGNORECASE)


def _strip_uuid_prefix(name: str) -> str:
    """Strip leading '<32-hex>_' UUID prefix and common file extensions.

    '1fd2be19fb0843eb9ce089d3da39087f_Handbuch EC3 Band 3.pdf'
      → 'Handbuch EC3 Band 3'
    """
    name = _UUID_PREFIX_RE.sub('', name)
    name = re.sub(r'\.(pdf|json|md)$', '', name, flags=re.IGNORECASE)
    return name.strip()


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
        "caption":     (human_text or annotation),
        "description": human_text,
        "image_type":  fig.get("type", "image"),
        "annotation":  annotation,
        "image_path":  img_url,
    }


def _extract_formula_context(
    paragraphs: List[Dict], formula_page: Any,
) -> str:
    """Extract paragraphs surrounding a formula on the same page as context.

    Returns up to ``settings.formula_context_chars`` characters of paragraph
    text from the same page, giving the LLM the prose that defines the
    formula's symbols and usage.
    """
    if not formula_page or not paragraphs:
        return ""

    frm_page_str = str(formula_page)
    same_page = [
        p.get("text", "") for p in paragraphs
        if str(p.get("page", "")) == frm_page_str and p.get("text")
    ]
    if not same_page:
        return ""

    joined = " ".join(same_page)
    max_chars = settings.formula_context_chars
    if len(joined) > max_chars:
        joined = joined[:max_chars]
    return joined


def _detect_chapter_type(ch: dict) -> str:
    """Classify a chapter as table_of_contents, introductory, appendix, main_chapter, or other."""
    title_lower = ch.get("title", "").lower()
    number = ch.get("number", "")

    toc_kws = ["inhaltsverzeichnis", "table of contents", "inhalt", "contents"]
    intro_kws = [
        "vorwort", "einleitung", "einführung", "vorbemerkung", "danksagung",
        "preface", "foreword", "introduction", "acknowledgement", "abstract",
        "zusammenfassung", "anmerkung",
    ]
    appendix_kws = ["anhang", "appendix", "anlage", "annex"]

    if any(k in title_lower for k in toc_kws):
        return "table_of_contents"
    if any(k in title_lower for k in intro_kws):
        return "introductory"
    if any(k in title_lower for k in appendix_kws):
        return "appendix"
    if number and (
        number.upper().startswith(("A.", "B.", "C.", "D.", "E.", "NA.", "NDP", "NCI"))
        or (len(number) > 0 and not number[0].isdigit())
    ):
        return "appendix"
    if number and number[0].isdigit():
        return "main_chapter"
    return "other"


def _detect_volumes(chapters: List[Dict], doc_name: str = "") -> List[Dict]:
    """Detect the volume for a document from its filename, wrapping all chapters in it.

    Volume information is encoded in the document filename, not in chapter headings.
    Supported patterns (case-insensitive):
    - "Band N"            → Handbuch EC3 Band 3, normen handbuch eurocode 8 band 2
    - "Band N und M"      → Handbuch EC 7 Band 1 und 2
    - "Anlage_N[_M]..."   → BEM-ING-Anlage_4_1_zum_ARS_22_2012-Entwurf (→ Anlage 4.1)

    If no pattern is matched, a single default volume "Band 1" is used.
    """
    # Pattern 1: "Band N und M" (multi-band label, e.g. "Band 1 und 2")
    m = re.search(r'\bband\s+(\d+)\s+und\s+(\d+)\b', doc_name, re.IGNORECASE)
    if m:
        label = f"Band {m.group(1)} und {m.group(2)}"
        number = m.group(1)
        return [{"title": label, "number": number, "chapters": chapters}]

    # Pattern 2: "Band N" (single band number)
    m = re.search(r'\bband\s+(\d+)\b', doc_name, re.IGNORECASE)
    if m:
        label = f"Band {m.group(1)}"
        return [{"title": label, "number": m.group(1), "chapters": chapters}]

    # Pattern 3: "Anlage_N[_M]" (BEM-ING annex files, e.g. Anlage_4_1 → Anlage 4.1)
    m = re.search(r'Anlage[_\s](\d+)(?:[_\s](\d+))?', doc_name, re.IGNORECASE)
    if m:
        number = m.group(1) + (f".{m.group(2)}" if m.group(2) else "")
        label = f"Anlage {number}"
        return [{"title": label, "number": number, "chapters": chapters}]

    # Fallback: single default volume
    return [{"title": "Band 1", "number": "1", "chapters": chapters}]



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
#  GraphBuilder
# ===================================================================== #


class GraphBuilder:
    """Build a Neo4j knowledge graph following the Graph-RAG schema."""

    def __init__(self):
        self.db = get_neo4j_connection()
        self._embedding_dim: Optional[int] = None

    # ----------------------------------------------------------------- #
    #  Embedding helpers
    # ----------------------------------------------------------------- #

    def _detect_embedding_dimensions(self) -> int:
        """Auto-detect the embedding model's output dimensions.

        Embeds a short probe text once and caches the result so that
        vector indexes are always created with the correct dimensionality,
        regardless of which embedding model is configured.
        """
        if self._embedding_dim is not None:
            return self._embedding_dim
        from backend.app.modules.ollama_client import get_ollama_client
        probe = get_ollama_client().generate_embedding("dimension probe")
        if not probe:
            raise RuntimeError(
                "Cannot detect embedding dimensions — embedding model returned "
                "an empty vector.  Is the Ollama embedding model loaded?"
            )
        self._embedding_dim = len(probe)
        logger.info("Detected embedding dimensions: %d", self._embedding_dim)
        return self._embedding_dim

    def _embed_chunked(
        self,
        ollama,
        text: str,
        strategy: str = "mean",
    ) -> list:
        """Embed *text* by splitting into overlapping chunks and combining.

        This avoids hardcoded truncation — the full text is embedded regardless
        of length.  Short texts that fit in a single chunk skip the splitting
        overhead entirely.

        Parameters
        ----------
        ollama : OllamaClient
            Client used to call the embedding model.
        text : str
            Full text to embed (any length).
        strategy : ``"mean"`` | ``"weighted"``
            ``"mean"``     — equal-weight average of all chunk embeddings.
            ``"weighted"`` — first chunk gets 2× weight (useful when the
            beginning of the text carries the most important information,
            e.g. title + intro for sections and chapters).

        Returns
        -------
        list[float]
            Combined embedding vector, or ``[]`` on failure.
        """
        chunk_size = settings.embedding_chunk_size
        overlap = settings.embedding_chunk_overlap

        # Guard: overlap must be strictly less than chunk_size
        if overlap >= chunk_size:
            overlap = 0

        # Fast path: text fits in a single chunk
        if len(text) <= chunk_size:
            return ollama.generate_embedding(text)

        # Split into overlapping chunks
        stride = chunk_size - overlap
        chunks: List[str] = []
        start = 0
        while start < len(text):
            chunks.append(text[start : start + chunk_size])
            start += stride

        # Embed each chunk
        embeddings: List[list] = []
        for chunk in chunks:
            emb = ollama.generate_embedding(chunk)
            if emb:
                embeddings.append(emb)

        if not embeddings:
            return []
        if len(embeddings) == 1:
            return embeddings[0]

        # Combine with the chosen strategy
        dim = len(embeddings[0])
        if strategy == "weighted":
            # First chunk gets 2× weight
            weights = [2.0] + [1.0] * (len(embeddings) - 1)
        else:
            weights = [1.0] * len(embeddings)

        total_weight = sum(weights)
        combined = [0.0] * dim
        for emb, w in zip(embeddings, weights):
            for i in range(dim):
                combined[i] += emb[i] * w
        for i in range(dim):
            combined[i] /= total_weight

        return combined

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
            "formulas": 0,
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
        }

        # ── Document ────────────────────────────────────────────────────
        # doc_id  = exact filename including UUID prefix (unique stable key)
        # doc_name = human-readable name stripped of UUID prefix and extension
        filename = data.get("source_file", os.path.basename(source_path))
        doc_id   = filename
        doc_name = _strip_uuid_prefix(filename)
        language = data.get("language", "de")
        doc_type = _detect_document_type(filename)
        # Use first page content for better Eurocode part detection
        _pages = data.get("pages") or []
        _first_page_content = (_pages[0].get("content", "") if _pages else "")[:3000]
        ec_part = _detect_eurocode_part(doc_name, _first_page_content)
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
                "id": doc_id, "filename": doc_name,
                "doc_type": doc_type, "ec_part": ec_part,
                "language": language,
                "version": data.get("version", "1.0"),
                "checksum": checksum,
            },
        )
        stats["documents"] = 1

        # ── Page nodes ───────────────────────────────────────────────────
        # Use JSON "pages" field (set by OCR pipeline); fall back to raw page_data arg
        page_id_map: Dict[int, str] = {}
        pages_list = data.get("pages")
        if not pages_list and page_data:
            pages_list = [
                {
                    "page_num": pd["page_num"],
                    "content": pd.get("markdown", ""),
                    "header": pd.get("header", ""),
                    "footer": pd.get("footer", ""),
                }
                for pd in page_data
            ]
        if pages_list:
            page_id_map = self._create_pages(doc_name, pages_list)
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

        volumes = _detect_volumes(raw_chapters, doc_name)
        for vol_data in volumes:
            vol_id = self._create_volume(doc_id, vol_data)
            stats["volumes"] += 1

            for ch in vol_data["chapters"]:
                ch_title = ch.get("title", "")
                ch_number = ch.get("number", str(len(chapter_id_map) + 1))
                ch_id = _make_uuid("chapter", doc_name, ch_number)
                chapter_id_map[ch_number] = ch_id

                ch_type = _detect_chapter_type(ch)

                self.db.execute_query(
                    """MERGE (ch:Chapter {id: $id})
                       SET ch.number       = $number,
                           ch.title        = $title,
                           ch.chapter_type = $ch_type,
                           ch.start_page   = $start_page,
                           ch.end_page     = $end_page
                       WITH ch
                       MATCH (d:Document {id: $doc_id})
                       MERGE (d)-[:HAS_CHAPTER]->(ch)
                       WITH ch
                       MATCH (v:Volume {id: $vol_id})
                       MERGE (v)-[:HAS_CHAPTER]->(ch)""",
                    {
                        "id": ch_id, "number": ch_number, "title": ch_title,
                        "ch_type": ch_type,
                        "start_page": ch.get("start_page", 0),
                        "end_page": ch.get("end_page", 0),
                        "doc_id": doc_id,
                        "vol_id": vol_id,
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
                    "full_text": full_text,
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
                        ),
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
                # Extract surrounding paragraph text from the same page
                frm_context = _extract_formula_context(paragraphs, frm.get("page"))
                self.db.execute_query(
                    """MERGE (f:Formula {id: $id})
                       SET f.latex         = $latex,
                           f.unicode       = $unicode,
                           f.section_title = $section_title,
                           f.context       = $context,
                           f.page          = $page,
                           f.embedding     = null
                       WITH f
                       MATCH (s:Section {id: $sid})
                       MERGE (s)-[:HAS_FORMULA]->(f)""",
                    {
                        "id": frm_id,
                        "latex": raw_latex,
                        "unicode": unicode_formula,
                        "section_title": sec_title,
                        "context": frm_context,
                        "page": frm.get("page"),
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

        logger.info("Ingested document '%s': %s", doc_name, stats)
        return stats

    # ----------------------------------------------------------------- #
    #  Page creation
    # ----------------------------------------------------------------- #

    def _create_volume(self, doc_id: str, vol: dict) -> str:
        """Create or update a Volume node and link it to the Document."""
        vid = _make_uuid("volume", doc_id, vol["title"])
        self.db.execute_query(
            """MERGE (v:Volume {id: $id})
               SET v.title  = $title,
                   v.number = $number
               WITH v
               MATCH (d:Document {id: $doc_id})
               MERGE (d)-[:HAS_VOLUME]->(v)""",
            {"id": vid, "title": vol["title"], "number": vol["number"], "doc_id": doc_id},
        )
        return vid

    def _create_pages(
        self, doc_name: str, page_data: List[Dict[str, Any]],
    ) -> Dict[int, str]:
        """Create Page nodes and NEXT_PAGE chain.  Return {page_num: page_id}."""
        page_id_map: Dict[int, str] = {}
        for pd in page_data:
            pnum = pd["page_num"]
            pid = _make_uuid("page", doc_name, str(pnum))
            page_id_map[pnum] = pid

            # Support both "content" (new) and "markdown" (legacy OCR) keys
            content = pd.get("content") or pd.get("markdown", "")
            header = pd.get("header", "")
            footer = pd.get("footer", "")
            if not header and content:
                header = content.split("\n")[0].strip()[:200]
            if not footer and content:
                lines = content.split("\n")
                footer = lines[-1].strip()[:200] if len(lines) > 1 else ""

            self.db.execute_query(
                """MERGE (p:Page {id: $id})
                   SET p.page_number = $pnum,
                       p.header      = $header,
                       p.footer      = $footer,
                       p.content     = $content,
                       p.embedding   = null""",
                {
                    "id": pid, "pnum": pnum,
                    "header": header, "footer": footer,
                    "content": content,
                },
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
            "CREATE INDEX page_number_idx IF NOT EXISTS FOR (p:Page) ON (p.page_number)",
            "CREATE INDEX doc_type_idx IF NOT EXISTS FOR (d:Document) ON (d.document_type)",
        ]
        for q in indexes:
            try:
                self.db.execute_query(q)
            except Exception as e:
                logger.debug("Index may already exist: %s", e)

        # Drop fulltext indexes before recreating so new fields are included
        for idx_name in ("page_fulltext", "formula_fulltext"):
            try:
                self.db.execute_query(f"DROP INDEX {idx_name} IF EXISTS")
            except Exception:
                pass

        fulltext_indexes = [
            """CREATE FULLTEXT INDEX section_fulltext IF NOT EXISTS
               FOR (s:Section) ON EACH [s.title, s.full_text, s.content_preview]""",
            """CREATE FULLTEXT INDEX table_fulltext IF NOT EXISTS
               FOR (t:Table) ON EACH [t.caption, t.content, t.section_title]""",
            """CREATE FULLTEXT INDEX formula_fulltext IF NOT EXISTS
               FOR (f:Formula) ON EACH [f.latex, f.unicode, f.section_title, f.context]""",
            """CREATE FULLTEXT INDEX figure_fulltext IF NOT EXISTS
               FOR (f:Figure) ON EACH [f.caption, f.description, f.annotation]""",
            """CREATE FULLTEXT INDEX page_fulltext IF NOT EXISTS
               FOR (p:Page) ON EACH [p.content, p.header, p.footer]""",
        ]
        for q in fulltext_indexes:
            try:
                self.db.execute_query(q)
            except Exception:
                pass

    # ----------------------------------------------------------------- #
    #  Rich summary generation (Chapter + Document)
    # ----------------------------------------------------------------- #

    _CHAPTER_SUMMARY_SYSTEM = (
        "You are a technical summarizer for Eurocode structural engineering "
        "documents.  Given the chapter structure below (section titles, section "
        "content, formula symbols, and table captions), write a concise "
        "{max_words}-word description of what this chapter covers.  Focus on "
        "the technical topics, standards referenced, and key formulas/tables.  "
        "Write in the same language as the source content.  "
        "Return ONLY the description — no labels, no headers, no bullet points."
    )

    def _extract_formula_symbols(self, expressions: List[str]) -> List[str]:
        """Extract single-character formula symbols from LaTeX/Unicode expressions."""
        symbols: set = set()
        # Greek letters and common engineering symbols
        greek = {
            "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
            "epsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ",
            "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ",
            "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
            "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ",
            "chi": "χ", "psi": "ψ", "omega": "ω",
            "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
            "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ",
            "Psi": "Ψ", "Omega": "Ω",
        }
        for expr in expressions:
            if not expr:
                continue
            # Extract from LaTeX \alpha, \gamma, etc.
            for name, char in greek.items():
                if f"\\{name}" in expr:
                    symbols.add(char)
            # Extract standalone Unicode Greek characters already in the text
            for char in greek.values():
                if char in expr:
                    symbols.add(char)
        return sorted(symbols)

    def _build_chapter_summary(self, chapter_id: str, ollama) -> str:
        """Build a rich summary for a single chapter from its graph data.

        Queries all sections (full text), formula symbols, and table captions,
        then calls the LLM for a concise description.
        """
        # Get chapter metadata
        ch_rows = self.db.execute_query(
            """MATCH (ch:Chapter {id: $id})
               RETURN ch.number AS number, ch.title AS title""",
            {"id": chapter_id},
        )
        if not ch_rows:
            return ""
        ch = ch_rows[0]
        ch_number = ch.get("number", "")
        ch_title = ch.get("title", "")

        # Get all sections with full text
        sections = self.db.execute_query(
            """MATCH (ch:Chapter {id: $id})-[:HAS_SECTION]->(s:Section)
               RETURN s.title AS title, s.full_text AS full_text,
                      s.content_preview AS preview
               ORDER BY s.number""",
            {"id": chapter_id},
        ) or []

        # Get formula expressions for symbol extraction
        formula_rows = self.db.execute_query(
            """MATCH (ch:Chapter {id: $id})-[:HAS_SECTION]->(:Section)
                     -[:HAS_FORMULA]->(f:Formula)
               RETURN coalesce(f.unicode, f.latex) AS expr""",
            {"id": chapter_id},
        ) or []
        expressions = [r.get("expr", "") for r in formula_rows]
        symbols = self._extract_formula_symbols(expressions)

        # Get tables with numbers and captions
        table_rows = self.db.execute_query(
            """MATCH (ch:Chapter {id: $id})-[:HAS_SECTION]->(:Section)
                     -[:HAS_TABLE]->(t:Table)
               WHERE t.caption IS NOT NULL AND t.caption <> ''
               RETURN DISTINCT t.number AS number, t.caption AS caption""",
            {"id": chapter_id},
        ) or []

        # Get figures with numbers and captions
        figure_rows = self.db.execute_query(
            """MATCH (ch:Chapter {id: $id})-[:HAS_SECTION]->(:Section)
                     -[:HAS_FIGURE]->(f:Figure)
               WHERE f.caption IS NOT NULL AND f.caption <> ''
               RETURN DISTINCT f.number AS number, f.caption AS caption""",
            {"id": chapter_id},
        ) or []

        # ── Assemble structured text ────────────────────────────────────
        parts: List[str] = [f"Kapitel {ch_number}: {ch_title}"]
        parts.append("")

        if sections:
            parts.append("Abschnitte:")
            for sec in sections:
                title = sec.get("title", "")
                full_text = sec.get("full_text", "") or sec.get("preview", "") or ""
                # First paragraph for the summary_text field
                first_para = full_text.split("\n\n")[0].strip() if full_text else ""
                if first_para:
                    parts.append(f"- {title}: {first_para}")
                else:
                    parts.append(f"- {title}")
            parts.append("")

        if symbols:
            parts.append(f"Formelsymbole: {', '.join(symbols)}")
            parts.append("")

        if table_rows:
            parts.append("Tabellen:")
            for tbl in table_rows:
                num = tbl.get("number") or ""
                cap = tbl.get("caption") or ""
                if num and cap:
                    parts.append(f"- Tabelle {num}: {cap}")
                elif cap:
                    parts.append(f"- {cap}")
            parts.append("")

        if figure_rows:
            parts.append("Abbildungen:")
            for fig in figure_rows:
                num = fig.get("number") or ""
                cap = fig.get("caption") or ""
                if num and cap:
                    parts.append(f"- Bild {num}: {cap}")
                elif cap:
                    parts.append(f"- {cap}")
            parts.append("")

        structured_text = "\n".join(parts)

        # ── LLM description from full section texts ─────────────────────
        llm_input_parts: List[str] = []
        for sec in sections:
            title = sec.get("title", "")
            full_text = sec.get("full_text", "") or ""
            if full_text:
                llm_input_parts.append(f"## {title}\n{full_text}")
            else:
                llm_input_parts.append(f"## {title}")
        llm_input = "\n\n".join(llm_input_parts)

        llm_description = ""
        if llm_input.strip():
            system_prompt = self._CHAPTER_SUMMARY_SYSTEM.format(
                max_words=settings.summary_max_words
            )
            try:
                llm_description = ollama.generate_text(
                    f"{system_prompt}\n\n---\n\n{llm_input}",
                    temperature=0.3,
                )
            except Exception as e:
                logger.warning(
                    "LLM summary failed for chapter %s: %s", chapter_id, e
                )

        if llm_description:
            structured_text += f"Zusammenfassung:\n{llm_description.strip()}"

        return structured_text.strip()

    def _build_document_summary(self, doc_id: str) -> str:
        """Build a document summary by concatenating all chapter summaries."""
        doc_rows = self.db.execute_query(
            """MATCH (d:Document {id: $id})
               RETURN d.filename AS filename,
                      d.eurocode_part AS eurocode_part,
                      d.document_type AS document_type""",
            {"id": doc_id},
        )
        if not doc_rows:
            return ""
        doc = doc_rows[0]
        filename = doc.get("filename", "")
        ec_part = doc.get("eurocode_part", "")
        doc_type = doc.get("document_type", "")

        chapter_rows = self.db.execute_query(
            """MATCH (d:Document {id: $id})-[:HAS_CHAPTER]->(ch:Chapter)
               WHERE ch.summary_text IS NOT NULL AND ch.summary_text <> ''
               RETURN ch.summary_text AS summary
               ORDER BY ch.number""",
            {"id": doc_id},
        ) or []

        header = f"Dokument: {filename}"
        if ec_part or doc_type:
            meta = ", ".join(p for p in [ec_part, doc_type] if p)
            header += f" ({meta})"

        chapter_summaries = [r["summary"] for r in chapter_rows if r.get("summary")]
        if not chapter_summaries:
            return header

        return header + "\n\n" + "\n\n".join(chapter_summaries)

    def generate_summaries(self) -> int:
        """Generate rich summaries for all Chapter and Document nodes.

        Must run after ``ingest_document()`` (so sections, formulas, tables
        exist) and before ``generate_embeddings()`` (so summaries can be
        embedded).

        Returns total number of summaries created.
        """
        from backend.app.modules.ollama_client import get_ollama_client
        ollama = get_ollama_client()
        count = 0

        # ── Chapter summaries ───────────────────────────────────────────
        chapters = self.db.execute_query(
            """MATCH (ch:Chapter)
               WHERE ch.summary_text IS NULL
               RETURN ch.id AS id, ch.title AS title""",
        ) or []

        logger.info("Generating summaries for %d chapters …", len(chapters))
        for ch in chapters:
            try:
                summary = self._build_chapter_summary(ch["id"], ollama)
                if summary:
                    self.db.execute_query(
                        """MATCH (ch:Chapter {id: $id})
                           SET ch.summary_text = $summary""",
                        {"id": ch["id"], "summary": summary},
                    )
                    count += 1
                    logger.debug("Summary for chapter '%s': %d chars",
                                 ch.get("title", ""), len(summary))
            except Exception as e:
                logger.warning("Summary failed for chapter %s: %s", ch["id"], e)

        # ── Document summaries ──────────────────────────────────────────
        docs = self.db.execute_query(
            """MATCH (d:Document)
               RETURN d.id AS id, d.filename AS filename""",
        ) or []

        logger.info("Generating summaries for %d documents …", len(docs))
        for doc in docs:
            try:
                summary = self._build_document_summary(doc["id"])
                if summary:
                    self.db.execute_query(
                        """MATCH (d:Document {id: $id})
                           SET d.summary_text = $summary""",
                        {"id": doc["id"], "summary": summary},
                    )
                    count += 1
                    logger.debug("Summary for document '%s': %d chars",
                                 doc.get("filename", ""), len(summary))
            except Exception as e:
                logger.warning("Summary failed for document %s: %s", doc["id"], e)

        logger.info("Generated %d summaries", count)
        return count

    # ----------------------------------------------------------------- #
    #  Embedding generation
    # ----------------------------------------------------------------- #

    def generate_embeddings(self, doc_name: Optional[str] = None) -> int:
        """Generate embeddings for all node types via Ollama.

        Creates vector indexes (with auto-detected dimensions) if they
        don't exist.  Uses chunked embedding so the **full text** of every
        node is embedded — no hardcoded truncation.

        Returns total number of embeddings stored.
        """
        from backend.app.modules.ollama_client import get_ollama_client
        ollama = get_ollama_client()
        embedded = 0

        dim = self._detect_embedding_dimensions()

        # Ensure vector indexes — dimensions auto-detected from the model
        index_defs = [
            ("section_embedding_index", "Section", "s"),
            ("concept_embedding_index", "Concept", "c"),
            ("formula_embedding_index", "Formula", "f"),
            ("table_embedding_index", "Table", "t"),
            ("figure_embedding_index", "Figure", "f"),
            ("page_embedding_index", "Page", "p"),
            ("chapter_embedding_index", "Chapter", "ch"),
            ("document_embedding_index", "Document", "d"),
        ]
        for idx_name, label, var in index_defs:
            try:
                self.db.execute_query(
                    f"""CREATE VECTOR INDEX {idx_name} IF NOT EXISTS
                        FOR ({var}:{label}) ON ({var}.embedding)
                        OPTIONS {{indexConfig: {{
                            `vector.dimensions`: $dim,
                            `vector.similarity_function`: 'cosine'
                        }}}}""",
                    {"dim": dim},
                )
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
            text = f"{title}\n\n{body}".strip()
            if not text or len(text.strip()) < 10:
                continue
            try:
                emb = self._embed_chunked(ollama, text, strategy="weighted")
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
                emb = self._embed_chunked(ollama, text, strategy="mean")
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
            sec_title = frm.get("section_title", "")
            expr      = frm.get("expr", "") or frm.get("latex", "")
            text = f"{sec_title}\n{expr}".strip() if sec_title else expr
            if len(text.strip()) < 3:
                continue
            try:
                emb = self._embed_chunked(ollama, text, strategy="mean")
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
                emb = self._embed_chunked(ollama, text, strategy="mean")
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
                emb = self._embed_chunked(ollama, text, strategy="mean")
                if emb:
                    self.db.execute_query(
                        """MATCH (f:Figure {id: $id}) SET f.embedding = $emb""",
                        {"id": fig["id"], "emb": emb},
                    )
                    embedded += 1
            except Exception as e:
                logger.warning("Figure embedding failed %s: %s", fig["id"], e)

        # ── Embed Pages ──────────────────────────────────────────────────
        embedded += self._generate_page_embeddings(ollama)

        # ── Embed Chapters ───────────────────────────────────────────────
        embedded += self._generate_chapter_embeddings(ollama)

        # ── Embed Documents ──────────────────────────────────────────────
        embedded += self._generate_document_embeddings(ollama)

        logger.info("Stored %d embeddings", embedded)
        return embedded

    def _generate_page_embeddings(self, ollama=None) -> int:
        """Generate embeddings for Page nodes that have content but no embedding."""
        if ollama is None:
            from backend.app.modules.ollama_client import get_ollama_client
            ollama = get_ollama_client()

        pages = self.db.execute_query(
            """MATCH (p:Page)
               WHERE p.embedding IS NULL AND p.content IS NOT NULL AND p.content <> ''
               RETURN p.id AS id, p.content AS content
               LIMIT 2000""",
        )

        logger.info("Embedding %d pages …", len(pages))
        embedded = 0
        for pg in pages:
            text = pg["content"]
            if len(text.strip()) < 10:
                continue
            try:
                emb = self._embed_chunked(ollama, text, strategy="mean")
                if emb:
                    self.db.execute_query(
                        """MATCH (p:Page {id: $id}) SET p.embedding = $emb""",
                        {"id": pg["id"], "emb": emb},
                    )
                    embedded += 1
            except Exception as e:
                logger.warning("Page embedding failed %s: %s", pg["id"], e)

        return embedded

    def _generate_chapter_embeddings(self, ollama=None) -> int:
        """Generate embeddings for Chapter nodes that have summary_text."""
        if ollama is None:
            from backend.app.modules.ollama_client import get_ollama_client
            ollama = get_ollama_client()

        chapters = self.db.execute_query(
            """MATCH (ch:Chapter)
               WHERE ch.embedding IS NULL
                 AND ch.summary_text IS NOT NULL AND ch.summary_text <> ''
               RETURN ch.id AS id, ch.summary_text AS text
               LIMIT 2000""",
        )

        logger.info("Embedding %d chapters …", len(chapters))
        embedded = 0
        for ch in chapters:
            text = ch["text"]
            if len(text.strip()) < 10:
                continue
            try:
                emb = self._embed_chunked(ollama, text, strategy="weighted")
                if emb:
                    self.db.execute_query(
                        """MATCH (ch:Chapter {id: $id}) SET ch.embedding = $emb""",
                        {"id": ch["id"], "emb": emb},
                    )
                    embedded += 1
            except Exception as e:
                logger.warning("Chapter embedding failed %s: %s", ch["id"], e)

        return embedded

    def _generate_document_embeddings(self, ollama=None) -> int:
        """Generate embeddings for Document nodes that have summary_text."""
        if ollama is None:
            from backend.app.modules.ollama_client import get_ollama_client
            ollama = get_ollama_client()

        docs = self.db.execute_query(
            """MATCH (d:Document)
               WHERE d.embedding IS NULL
                 AND d.summary_text IS NOT NULL AND d.summary_text <> ''
               RETURN d.id AS id, d.summary_text AS text
               LIMIT 500""",
        )

        logger.info("Embedding %d documents …", len(docs))
        embedded = 0
        for doc in docs:
            text = doc["text"]
            if len(text.strip()) < 10:
                continue
            try:
                emb = self._embed_chunked(ollama, text, strategy="mean")
                if emb:
                    self.db.execute_query(
                        """MATCH (d:Document {id: $id}) SET d.embedding = $emb""",
                        {"id": doc["id"], "emb": emb},
                    )
                    embedded += 1
            except Exception as e:
                logger.warning("Document embedding failed %s: %s", doc["id"], e)

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
