"""OCR Pipeline - Mistral OCR extraction, structured JSON creation, graph building & embedding.

Pipeline stages
---------------
1. Split PDF into ≤80-page chunks.
2. Run Mistral OCR on every chunk → combined Markdown.
3. Parse Markdown with **mistune** AST (not regex) into structured sections.
4. Save structured JSON + raw Markdown to backend/json/.
5. Ingest into Neo4j via GraphBuilder (Document, Chapter, Page, Section,
   Table, Figure, Formula, Concept nodes).
6. Generate embeddings on Section and Concept nodes via GraphBuilder.
7. Compute cross-document semantic similarity links.

Processing status is persisted in PostgreSQL (see processing_db module).
"""
from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import threading
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from mistralai.extra import response_format_from_pydantic_model
from enum import Enum
from pypdf import PdfReader, PdfWriter

from backend.app.modules.latex_utils import latex_to_unicode, normalize_latex

logger = logging.getLogger(__name__)

class ImageType(str, Enum):
    GRAPH = "graph"
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"

class Image(BaseModel):
    image_type: ImageType = Field(..., description="The type of the image. Must be one of 'graph', 'text', 'table' or 'image'.")
    description: str = Field(..., description="A description of the image.")


# ---------------------------------------------------------------------------
#  Status helpers (thin wrappers over processing_db)
# ---------------------------------------------------------------------------

def _set_status(
    filename: str,
    status: str,
    step: str,
    error: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> None:
    from backend.app.modules.processing_db import upsert_status
    upsert_status(
        filename=filename,
        status=status,
        step=step,
        error=error,
        stats=stats,
        started_at=started_at,
        completed_at=completed_at,
    )


def get_processing_status(filename: str) -> Optional[Dict[str, Any]]:
    """Public: return status dict for one document (may be None)."""
    from backend.app.modules.processing_db import get_status
    return get_status(filename)


def get_all_processing_statuses() -> List[Dict[str, Any]]:
    """Public: return all status rows as a list."""
    from backend.app.modules.processing_db import get_all_statuses
    return get_all_statuses()


# ---------------------------------------------------------------------------
#  PDF helpers
# ---------------------------------------------------------------------------

def _split_pdf_into_chunks(pdf_path: str, chunk_size: int = 80) -> List[str]:
    """Split a large PDF into temporary files of at most *chunk_size* pages."""
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    chunk_paths: List[str] = []

    logger.info("PDF %s — %d pages, splitting into chunks of %d", pdf_path, num_pages, chunk_size)

    for start in range(0, num_pages, chunk_size):
        end = min(start + chunk_size, num_pages)
        writer = PdfWriter()
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        writer.write(tmp)
        tmp.close()
        chunk_paths.append(tmp.name)

    logger.info("Created %d PDF chunk(s)", len(chunk_paths))
    return chunk_paths


def _encode_pdf_base64(pdf_path: str) -> str:
    with open(pdf_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


# ---------------------------------------------------------------------------
#  Mistral OCR
# ---------------------------------------------------------------------------

def _save_image_to_disk(
    image_b64: str,
    doc_stem: str,
    page_num: int,
    img_id: str,
) -> Optional[str]:
    """Save a base64-encoded image to backend/images/ and return the URL path.

    Returns None if saving fails.
    """
    if not image_b64:
        return None
    try:
        import re as _re
        # Strip data URI prefix if present: "data:image/png;base64,..."
        b64_data = _re.sub(r"^data:[^;]+;base64,", "", image_b64)
        img_bytes = base64.b64decode(b64_data)

        images_dir = Path(__file__).parent.parent.parent / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize img_id for use in filename
        safe_id = _re.sub(r"[^a-zA-Z0-9_\-]", "_", img_id)[:40]
        safe_doc = _re.sub(r"[^a-zA-Z0-9_\-]", "_", doc_stem)[:30]
        filename = f"{safe_doc}_p{page_num}_{safe_id}.png"
        file_path = images_dir / filename

        file_path.write_bytes(img_bytes)
        logger.debug("Saved image: %s (%d bytes)", file_path, len(img_bytes))
        return f"/api/images/{filename}"
    except Exception as e:
        logger.warning("Failed to save image %s: %s", img_id, e)
        return None


def _run_mistral_ocr(pdf_path: str, api_key: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Return *(combined_markdown, page_data_list)* for the PDF at *pdf_path*.

    Each element of ``page_data_list`` is:
        {"page_num": int, "markdown": str, "images": list}
    """
    from mistralai import Mistral

    client = Mistral(api_key=api_key)
    chunk_paths = _split_pdf_into_chunks(pdf_path, chunk_size=40)

    # Use the PDF stem for image filenames
    doc_stem = Path(pdf_path).stem[:30]

    all_md: List[str] = []
    all_pages: List[Dict[str, Any]] = []
    global_offset = 0

    try:
        for i, chunk_path in enumerate(chunk_paths):
            logger.info("OCR chunk %d/%d …", i + 1, len(chunk_paths))
            b64 = _encode_pdf_base64(chunk_path)

            resp = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{b64}",
                },
                bbox_annotation_format=response_format_from_pydantic_model(Image),
                include_image_base64=True,
                extract_header=True,
                extract_footer=True,
                table_format="markdown",
            )

            for idx, page in enumerate(resp.pages):
                page_num = global_offset + idx + 1
                page_md = page.markdown
                images_info: List[Dict[str, Any]] = []
                annotations: List[Dict[str, Any]] = []

                # Capture bbox annotations (image/table descriptions) when present
                for ann in getattr(page, "annotations", []) or getattr(page, "bbox_annotations", []):
                    annotations.append({
                        "id": getattr(ann, "id", ""),
                        "type": getattr(ann, "image_type", getattr(ann, "type", "")),
                        "description": getattr(ann, "description", ""),
                        "bbox": getattr(ann, "bbox", None),
                    })

                if hasattr(page, "images") and page.images:
                    for img in page.images:
                        img_id = getattr(img, "id", f"img_{len(images_info)}")
                        desc = getattr(img, "description", "") or img_id

                        # Save base64 image to disk if available
                        img_b64 = (
                            getattr(img, "image_base64", None)
                            or getattr(img, "base64", None)
                            or getattr(img, "data", None)
                        )
                        image_url = _save_image_to_disk(img_b64, doc_stem, page_num, img_id)

                        images_info.append({
                            "id": img_id,
                            "description": desc,
                            "image_url": image_url,
                        })

                        # Replace Markdown image syntax: embed URL if saved, else text
                        if image_url:
                            page_md = page_md.replace(
                                f"![{img_id}]({img_id})",
                                f"![{desc}]({image_url})",
                            )
                        else:
                            page_md = page_md.replace(
                                f"![{img_id}]({img_id})",
                                f"[Abbildung: {desc}]",
                            )

                # Inline table content — Mistral puts actual table data in
                # page.tables and only emits [tbl-N.md](tbl-N.md) placeholders
                # in page.markdown.  The id field already includes the extension
                # (e.g. "tbl-11.md"), so the placeholder is literally [{id}]({id}).
                for tbl in getattr(page, "tables", []) or []:
                    tbl_id = getattr(tbl, "id", None)
                    tbl_content = getattr(tbl, "content", None) or ""
                    if tbl_id and tbl_content:
                        page_md = page_md.replace(
                            f"[{tbl_id}]({tbl_id})", tbl_content
                        )

                all_pages.append({
                    "page_num": page_num,
                    "markdown": page_md,
                    "images": images_info,
                    "annotations": annotations,
                })
                all_md.append(page_md)

            global_offset += len(resp.pages)

    finally:
        for p in chunk_paths:
            if os.path.exists(p):
                os.unlink(p)

    return "\n\n".join(all_md), all_pages


# ---------------------------------------------------------------------------
#  Markdown → structured JSON   (mistune AST — no regex)
# ---------------------------------------------------------------------------

def _make_md_parser():
    """Build a mistune Markdown parser that returns an AST (token list)."""
    import mistune

    plugins: List[Any] = ["table"]
    # math plugin is available in mistune >= 3.0.2
    try:
        import mistune.plugins.math  # noqa: F401
        plugins.append("math")
    except ImportError:
        logger.error(
            "mistune math plugin unavailable — formula blocks ($$...$$, $...$) "
            "will NOT be parsed as math tokens and formulas will be lost. "
            "Install mistune >= 3.0.2 to fix this."
        )

    # renderer=None → returns token list instead of rendered HTML
    return mistune.create_markdown(renderer=None, plugins=plugins)


_MD_PARSER = None  # created once per process


def _get_md_parser():
    global _MD_PARSER
    if _MD_PARSER is None:
        _MD_PARSER = _make_md_parser()
    return _MD_PARSER


# ---- inline token helpers ------------------------------------------------

def _inline_text(children: Optional[List[Dict]]) -> str:
    """Recursively collapse inline token children into plain text."""
    if not children:
        return ""
    parts: List[str] = []
    for child in children:
        t = child.get("type", "")
        if t in ("text", "codespan"):
            parts.append(child.get("raw", ""))
        elif t == "softline":
            parts.append(" ")
        elif t == "linebreak":
            parts.append("\n")
        elif t == "image":
            parts.append(child.get("attrs", {}).get("alt", ""))
        elif "children" in child:
            parts.append(_inline_text(child["children"]))
        else:
            parts.append(child.get("raw", ""))
    return "".join(parts)


def _table_plain_text(token: Dict) -> str:
    """Extract all cell text from a table token as a flat string."""
    cells: List[str] = []
    for row in token.get("children", []):
        for cell in row.get("children", []):
            cells.append(_inline_text(cell.get("children")))
    return " | ".join(c for c in cells if c)


def _inline_images(children: Optional[List[Dict]]) -> List[str]:
    """Collect alt/title strings from image tokens in inline content."""
    if not children:
        return []
    descs: List[str] = []
    for child in children:
        if child.get("type") == "image":
            attrs = child.get("attrs", {})
            desc = attrs.get("alt") or attrs.get("title") or attrs.get("url", "")
            if desc:
                descs.append(desc)
        elif "children" in child:
            descs.extend(_inline_images(child["children"]))
    return descs


# ---- page estimation -----------------------------------------------------

def _estimate_page(text: str, page_data: List[Dict[str, Any]]) -> int:
    """Estimate the source page by word-overlap against each page's markdown."""
    if not text or not page_data:
        return 1
    words = set(text.split()[:15])
    best, best_score = 1, 0
    for pd in page_data:
        score = len(words & set(pd["markdown"].split()))
        if score > best_score:
            best_score, best = score, pd["page_num"]
    return best


# ---- annotation overlap helper -------------------------------------------

def _overlaps(a: str, b: str) -> bool:
    """Two-directional substring check for annotation deduplication.

    Handles paraphrased or abbreviated OCR descriptions (e.g. bbox label vs.
    in-text caption).  Returns True when either string is a substring of the
    other, or when their first 20 chars match (prefix heuristic for long
    strings).
    """
    d, c = a.lower().strip(), b.lower().strip()
    return bool(d and c and (d in c or c in d or (len(d) > 20 and d[:20] in c)))


# ---- main parser ---------------------------------------------------------

def _parse_markdown_to_structured(
    full_markdown: str,
    page_data: List[Dict[str, Any]],
    source_filename: str,
) -> Dict[str, Any]:
    """Parse OCR Markdown via the mistune AST into a structured document dict."""

    doc_name = Path(source_filename).stem
    tokens: List[Dict] = _get_md_parser()(full_markdown) or []

    # Pre-index annotations by page number for later attachment
    annotations_by_page = {
        pd.get("page_num"): pd.get("annotations", [])
        for pd in page_data or []
        if pd.get("annotations")
    }

    # ---- detect document language ----------------------------------------
    lower = full_markdown[:3000].lower()
    de_hits = sum(1 for w in {"der ", "die ", "das ", "und ", "ist ", "von "} if w in lower)
    en_hits = sum(1 for w in {"the ", "this ", "that ", "with ", "from "} if w in lower)
    lang = "de" if de_hits >= en_hits else "en"

    # ---- state -----------------------------------------------------------
    chapters: List[Dict[str, Any]] = []
    sections: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    # Accumulators for items not yet committed to a section
    pend_paras: List[str] = []
    pend_tables: List[Dict] = []
    pend_images: List[Dict] = []
    pend_formulas: List[Dict] = []

    def _new_section(title: str, level: int) -> Dict[str, Any]:
        return {
            "section": title, "level": level, "content": "",
            "paragraphs": [], "tables": [], "images": [], "formulas": [],
        }

    def _flush() -> None:
        """Push pending accumulators into the current section."""
        nonlocal pend_paras, pend_tables, pend_images, pend_formulas
        if current is None:
            return
        for p in pend_paras:
            if len(p) >= 5:
                current["paragraphs"].append(
                    {"text": p, "page": _estimate_page(p, page_data)}
                )
        current["tables"].extend(pend_tables)
        current["images"].extend(pend_images)
        current["formulas"].extend(pend_formulas)
        pend_paras.clear(); pend_tables.clear()
        pend_images.clear(); pend_formulas.clear()

    # ---- walk token list -------------------------------------------------
    for token in tokens:
        kind = token.get("type", "")

        # -- headings ---------------------------------------------------------
        if kind == "heading":
            level = token.get("attrs", {}).get("level", 2)
            title = _inline_text(token.get("children"))
            _flush()
            if current is not None:
                sections.append(current)
            current = _new_section(title, level)
            if level == 1:
                chapters.append({
                    "title": title,
                    "number": str(len(chapters) + 1),
                    "section_refs": [],
                })
            if chapters:
                chapters[-1]["section_refs"].append(title)

        # -- paragraphs -------------------------------------------------------
        elif kind == "paragraph":
            text = _inline_text(token.get("children")).strip()
            if text:
                pend_paras.append(text)
            for desc in _inline_images(token.get("children")):
                pend_images.append({
                    "id": f"img_{len(pend_images)+1}", "description": desc,
                    "type": "image",
                    "page": _estimate_page(desc, page_data),
                })

        # -- tables -----------------------------------------------------------
        elif kind == "table":
            txt = _table_plain_text(token)
            pend_tables.append({
                "id": f"table_{len(pend_tables)+1}",
                "text": txt,
                "caption": current["section"] if current else "",
                "page": _estimate_page(txt, page_data),
            })

        # -- block math ( $$...$$ ) -------------------------------------------
        elif kind in ("block_math", "math_block"):
            raw = normalize_latex(token.get("raw", "").strip())
            if raw:
                pend_formulas.append({
                    "expression": raw,
                    "unicode": latex_to_unicode(raw),
                    "context": current["section"] if current else "",
                    "page": _estimate_page(raw, page_data),
                })

        # -- inline math (standalone token) -----------------------------------
        elif kind == "inline_math":
            raw = normalize_latex(token.get("raw", "").strip())
            if raw and len(raw) >= 1:
                pend_formulas.append({
                    "expression": raw,
                    "unicode": latex_to_unicode(raw),
                    "context": current["section"] if current else "",
                    "page": _estimate_page(raw, page_data),
                })

        # -- fenced code (treat math-annotated blocks as formulas) ------------
        elif kind == "block_code":
            raw = token.get("raw", "").strip()
            info = (token.get("attrs") or {}).get("info", "")
            if info in ("math", "latex", "tex") and raw:
                raw = normalize_latex(raw)
                pend_formulas.append({
                    "expression": raw,
                    "unicode": latex_to_unicode(raw),
                    "context": current["section"] if current else "",
                    "page": _estimate_page(raw, page_data),
                })
            elif raw and len(raw) > 20:
                pend_paras.append(raw)

        # -- standalone image -------------------------------------------------
        elif kind == "image":
            attrs = token.get("attrs", {})
            desc = attrs.get("alt") or attrs.get("title") or attrs.get("url", "")
            if desc:
                pend_images.append({
                    "id": f"img_{len(pend_images)+1}", "description": desc,
                    "type": "image",
                    "page": _estimate_page(desc, page_data),
                })

        # -- lists (flatten to paragraph text) --------------------------------
        elif kind == "list":
            items: List[str] = []
            for item in token.get("children", []):
                for bc in item.get("children", []):
                    if bc.get("type") == "block_text":
                        t = _inline_text(bc.get("children"))
                        if t:
                            items.append(t)
            combined = " • ".join(items)
            if combined:
                pend_paras.append(combined)

        # -- everything else (hr, blank, etc.) → skip -------------------------

    # flush last section
    _flush()
    if current is not None:
        sections.append(current)
    elif pend_paras or pend_tables or pend_images or pend_formulas:
        # no headings at all
        fallback = _new_section("Inhalt", 1)
        current = fallback
        _flush()
        sections.append(fallback)

    # populate content field from first paragraphs
    for sec in sections:
        if sec["paragraphs"]:
            sec["content"] = " ".join(
                p["text"] for p in sec["paragraphs"][:3]
            )[:5000]

        # Attach bbox annotations (tables/images) that fall on this section's pages
        pages_for_sec = set()
        for p in sec.get("paragraphs", []):
            if p.get("page"):
                pages_for_sec.add(p["page"])
        for t in sec.get("tables", []):
            if t.get("page"):
                pages_for_sec.add(t["page"])
        for img in sec.get("images", []):
            if img.get("page"):
                pages_for_sec.add(img["page"])
        for frm in sec.get("formulas", []):
            if frm.get("page"):
                pages_for_sec.add(frm["page"])

        for pg in pages_for_sec:
            for ann in annotations_by_page.get(pg, []):
                desc = (ann.get("description") or "").strip()
                if not desc:
                    continue
                atype = (ann.get("type") or ann.get("image_type") or "").lower()

                if "table" in atype:
                    if not any(
                        _overlaps(desc, t.get("caption", "") or t.get("text", ""))
                        for t in sec.get("tables", [])
                    ):
                        sec.setdefault("tables", []).append({
                            "id": f"bbox_table_{len(sec.get('tables', []))+1}",
                            "text": desc,
                            "caption": desc,
                            "page": pg,
                            "annotation": desc,
                            "type": atype or "table",
                        })
                else:
                    if not any(
                        _overlaps(desc, i.get("description", ""))
                        for i in sec.get("images", [])
                    ):
                        sec.setdefault("images", []).append({
                            "id": f"bbox_img_{len(sec.get('images', []))+1}",
                            "description": desc,
                            "type": atype or "image",
                            "page": pg,
                            "annotation": desc,
                        })

    # collect all formulas for top-level key_formulas
    all_formulas: List[Dict] = [
        {
            "name": f.get("context") or f["expression"][:80],
            "formula": normalize_latex(f["expression"]),
            "unicode": f.get("unicode", latex_to_unicode(f["expression"])),
            "variables": {},
        }
        for sec in sections
        for f in sec.get("formulas", [])
    ]

    return {
        "document": doc_name,
        "source_file": source_filename,
        "language": lang,
        "ocr_processed": True,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "chapters": chapters,
        "sections": sections,
        "key_formulas": all_formulas,
        "references": [],
        "full_markdown": full_markdown,
    }


# ---------------------------------------------------------------------------
#  Main pipeline
# ---------------------------------------------------------------------------

def process_document(pdf_path: str, filename: str) -> Dict[str, Any]:
    """Full pipeline: OCR → structured JSON → Neo4j graph → embeddings.

    Processing status is updated in PostgreSQL at every stage so the
    front-end can poll ``GET /api/documents/processing-status/<filename>``.
    """
    from config.settings import settings

    started_at = datetime.now(timezone.utc).isoformat()
    _set_status(filename, "processing", "starting", started_at=started_at)

    try:
        # 1. OCR
        _set_status(filename, "processing", "ocr_extraction", started_at=started_at)
        api_key = settings.mistral_api_key or os.getenv("MISTRAL_API_KEY", "")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is not configured.")

        combined_md, page_data = _run_mistral_ocr(pdf_path, api_key)
        logger.info("OCR done: %d chars, %d pages", len(combined_md), len(page_data))

        # 2. Parse Markdown → structured dict
        _set_status(filename, "processing", "parsing", started_at=started_at)
        structured = _parse_markdown_to_structured(combined_md, page_data, filename)

        # 3. Save JSON + Markdown
        _set_status(filename, "processing", "saving_json", started_at=started_at)
        json_dir = Path(__file__).parent.parent.parent / "json"
        json_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(filename).stem
        json_path = json_dir / f"{stem}.json"
        md_path   = json_dir / f"{stem}.md"

        # Add per-page OCR content for graph builder (page-level embeddings)
        structured["pages"] = [
            {
                "page_num": pd["page_num"],
                "content": pd.get("markdown", ""),
                "header": pd.get("header", ""),
                "footer": pd.get("footer", ""),
            }
            for pd in page_data
        ]

        # Omit full_markdown from the JSON file (too large); save as .md separately
        json_data = {k: v for k, v in structured.items() if k != "full_markdown"}
        json_path.write_text(
            json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        md_path.write_text(combined_md, encoding="utf-8")
        logger.info("Saved %s and %s", json_path, md_path)

        # 4. Build Neo4j graph
        _set_status(filename, "processing", "building_graph", started_at=started_at)
        from backend.app.modules.graph_builder import get_graph_builder
        builder = get_graph_builder()
        graph_stats = builder.ingest_document(structured, page_data=page_data)
        logger.info("Graph stats: %s", graph_stats)

        # 5. Embeddings on Section + Concept nodes, then cross-document similarity
        _set_status(filename, "processing", "generating_embeddings", started_at=started_at)
        embed_count = builder.generate_embeddings(structured.get("document", filename))
        similarity_count = builder.compute_semantic_similarity()

        # Done
        final_stats = {
            "pages": len(page_data),
            "characters": len(combined_md),
            "sections": len(structured.get("sections", [])),
            "chapters": len(structured.get("chapters", [])),
            "embeddings": embed_count,
            "semantic_links": similarity_count,
            "graph": graph_stats,
        }
        _set_status(
            filename, "completed", "done",
            stats=final_stats,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info("Pipeline completed for %s: %s", filename, final_stats)
        return final_stats

    except Exception as exc:
        logger.error("Pipeline failed for %s: %s", filename, exc, exc_info=True)
        _set_status(
            filename, "failed", "error",
            error=str(exc),
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        raise


def process_document_background(pdf_path: str, filename: str) -> None:
    """Fire-and-forget: run the pipeline in a daemon thread."""

    def _run() -> None:
        try:
            process_document(pdf_path, filename)
        except Exception as exc:
            logger.debug("Background thread finished with error for %s: %s", filename, exc)

    threading.Thread(
        target=_run, name=f"ocr-pipeline-{filename}", daemon=True
    ).start()
    logger.info("Background OCR pipeline started for %s", filename)
