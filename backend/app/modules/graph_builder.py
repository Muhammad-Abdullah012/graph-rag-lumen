"""Graph Building Module — Document, Page, Reference, Table, and Image nodes"""
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

from backend.app.modules.database import get_neo4j_connection
from backend.app.modules.ollama_client import get_ollama_client
from config.settings import settings

logger = logging.getLogger(__name__)

# Matches German/European engineering norm references, e.g.:
#   DIN EN 1993-2:2010-12, 9.5.2(3) bis (6)
#   EN 1992-1-1:2011-01 11
#   DIN EN 1993-1-9:2010-12, Abschnitt 6
#   EN 1993-1-1/NA:2015-08 NDP zu 6.3.2.2(2)
NORM_REF_PATTERN = re.compile(
    r'(?:(?:DIN|ISO|prEN|CEN)\s+)?'           # optional prefix
    r'(EN\s+\d{3,4}(?:-\d+)+(?:/[A-Z]+)?)'   # norm number: EN 1993-2, EN 1993-1-1/NA
    r'(?::\d{4}(?:-\d+)?)?'                   # optional date: :2010-12
    r'(?:'                                     # optional section reference
        r'\s*[,;]\s*'
        r'(?:(?:Abschnitt|Anhang|Bild|Tabelle|Gleichung|NDP zu)\s+)?'
        r'[\d.]+(?:\([^)]*\))?'               # section number: 9.5.2(3)
        r'(?:\s*(?:bis|und|–|-)\s*\(?[\d.]+(?:\([^)]*\))?\)?)?'  # optional range
    r')?',
    re.IGNORECASE,
)

# Caption patterns for tables and figures in German technical documents
TABLE_CAPTION_PATTERN = re.compile(
    r'(?:Tabelle|Tab\.?)\s+[\w.]+[^.\n]*',
    re.IGNORECASE,
)
IMAGE_CAPTION_PATTERN = re.compile(
    r'(?:Bild|Abb\.?|Abbildung|Fig\.?)\s+[\w.]+[^.\n]*',
    re.IGNORECASE,
)


class GraphBuilder:
    """Build a Neo4j graph with Document, Page, Reference, Table, and Image nodes."""

    def __init__(self):
        self.db = get_neo4j_connection()
        self.ollama = get_ollama_client()

    def build_graph(
        self,
        pages: list[dict],
        document_id: str,
        document_name: str,
        document_url: str,
    ):
        """
        Create Document + Page nodes and all sub-nodes (Reference, Table, Image).

        Args:
            pages: List of page dicts from MistralOCRPipeline.process()
            document_id: Unique document identifier
            document_name: Original filename (without UUID prefix)
            document_url: Relative URL to the PDF
        """
        logger.info(f"Building graph for document: {document_name} ({len(pages)} pages)")

        self._create_document_node(
            document_id=document_id,
            name=document_name,
            source_file=document_name,
            url=document_url,
            page_count=len(pages),
        )

        for page in pages:
            page_index = page.get("index", 0)
            page_id = f"{document_id}_p{page_index}"
            markdown = page.get("markdown") or ""

            self._create_page_node(page, document_id)

            if markdown.strip():
                refs = self._extract_norm_references(markdown, page_id, document_id)
                self._create_reference_nodes(refs)
                if refs:
                    self._resolve_references(refs)

            tables = page.get("tables") or []
            if tables:
                self._create_table_nodes(tables, markdown, page_id, document_id)

            images = [
                {k: v for k, v in img.items() if k != "image_base64"}
                for img in (page.get("images") or [])
            ]
            if images:
                self._create_image_nodes(images, markdown, page_id, document_id)

        self._create_next_page_links(document_id, len(pages))
        logger.info(f"Graph built for document: {document_name}")

    # ------------------------------------------------------------------
    # Document / Page
    # ------------------------------------------------------------------

    def _create_document_node(
        self,
        document_id: str,
        name: str,
        source_file: str,
        url: str,
        page_count: int,
    ):
        query = """
            MERGE (d:Document {id: $id})
            SET d.name = $name,
                d.source_file = $source_file,
                d.url = $url,
                d.page_count = $page_count,
                d.processed_at = $processed_at
        """
        self.db.execute_query(query, {
            "id": document_id,
            "name": name,
            "source_file": source_file,
            "url": url,
            "page_count": page_count,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.debug(f"Created/updated Document node: {document_id}")

    def _create_page_node(self, page: dict, document_id: str):
        page_index = page.get("index", 0)
        page_id = f"{document_id}_p{page_index}"
        markdown = page.get("markdown") or ""

        images = [
            {k: v for k, v in img.items() if k != "image_base64"}
            for img in (page.get("images") or [])
        ]

        embedding = self.ollama.generate_embedding(markdown) if markdown.strip() else []

        query = """
            MATCH (d:Document {id: $document_id})
            MERGE (p:Page {id: $id})
            SET p.document_id   = $document_id,
                p.page_number   = $page_number,
                p.markdown      = $markdown,
                p.images_json   = $images_json,
                p.tables_json   = $tables_json,
                p.hyperlinks_json = $hyperlinks_json,
                p.header        = $header,
                p.footer        = $footer,
                p.dimensions_json = $dimensions_json,
                p.embedding     = $embedding
            MERGE (p)-[:BELONGS_TO]->(d)
        """
        self.db.execute_query(query, {
            "id": page_id,
            "document_id": document_id,
            "page_number": page_index,
            "markdown": markdown,
            "images_json": json.dumps(images, ensure_ascii=False),
            "tables_json": json.dumps(page.get("tables") or [], ensure_ascii=False),
            "hyperlinks_json": json.dumps(page.get("hyperlinks") or [], ensure_ascii=False),
            "header": page.get("header") or "",
            "footer": page.get("footer") or "",
            "dimensions_json": json.dumps(page.get("dimensions") or {}, ensure_ascii=False),
            "embedding": embedding,
        })
        logger.debug(f"Created/updated Page node: {page_id}")

    def _create_next_page_links(self, document_id: str, page_count: int):
        """Chain consecutive pages with NEXT_PAGE relationships."""
        if page_count < 2:
            return
        query = """
            MATCH (p1:Page {id: $id1})
            MATCH (p2:Page {id: $id2})
            MERGE (p1)-[:NEXT_PAGE]->(p2)
            MERGE (p2)-[:PREV_PAGE]->(p1)
        """
        for i in range(page_count - 1):
            self.db.execute_query(query, {
                "id1": f"{document_id}_p{i}",
                "id2": f"{document_id}_p{i + 1}",
            })
        logger.debug(f"Created NEXT_PAGE chain for document {document_id}")

    # ------------------------------------------------------------------
    # Reference nodes
    # ------------------------------------------------------------------

    def _extract_norm_references(
        self, markdown: str, page_id: str, document_id: str
    ) -> List[Dict[str, Any]]:
        """Extract norm cross-references from page markdown using regex."""
        ctx = settings.reference_context_chars
        refs = []
        seen_texts: set = set()

        for match in NORM_REF_PATTERN.finditer(markdown):
            full_ref = match.group(0).strip()
            if not full_ref or full_ref in seen_texts:
                continue
            seen_texts.add(full_ref)

            # Normalise: strip date suffix, keep prefix + number
            norm_raw = match.group(1)  # e.g. "EN 1993-2" or "EN 1993-1-1/NA"
            norm = re.sub(r'\s+', ' ', norm_raw).strip()

            # Extract section number if present (digits and dots after the comma)
            section_match = re.search(
                r'[,;]\s*(?:Abschnitt\s+|Anhang\s+)?(\d[\d.]*)', full_ref
            )
            section = section_match.group(1) if section_match else ""

            # Context window around the match
            start = max(0, match.start() - ctx)
            end = min(len(markdown), match.end() + ctx)
            context = markdown[start:end]

            refs.append({
                "id": f"{page_id}_ref_{uuid.uuid4().hex[:8]}",
                "page_id": page_id,
                "document_id": document_id,
                "norm": norm,
                "section": section,
                "full_reference": full_ref,
                "context": context,
            })

        return refs

    def _create_reference_nodes(self, refs: List[Dict[str, Any]]):
        """Persist Reference nodes with embeddings and link them to their parent Page."""
        query = """
            MERGE (r:Reference {id: $id})
            SET r.page_id       = $page_id,
                r.document_id   = $document_id,
                r.norm          = $norm,
                r.section       = $section,
                r.full_reference = $full_reference,
                r.context       = $context,
                r.embedding     = $embedding
            WITH r
            MATCH (p:Page {id: $page_id})
            MERGE (r)-[:CITED_ON]->(p)
        """
        for ref in refs:
            embedding = self.ollama.generate_embedding(ref["context"])
            self.db.execute_query(query, {**ref, "embedding": embedding})
        logger.debug(f"Created {len(refs)} Reference node(s) for page {refs[0]['page_id'] if refs else '?'}")

    def _resolve_references(self, refs: List[Dict[str, Any]]):
        """Try to link each Reference to the Page(s) that actually contain that section.

        Strategy: search Page.header (and markdown) for the section number using the
        fulltext index.  Only pages from other documents are candidates — a page does
        not resolve to itself.
        """
        threshold = settings.reference_resolve_score_threshold
        merge_query = """
            MATCH (r:Reference {id: $ref_id})
            MATCH (p:Page {id: $page_id})
            MERGE (r)-[:RESOLVED_TO]->(p)
        """
        for ref in refs:
            section = ref.get("section", "").strip()
            if not section:
                continue
            try:
                results = self.db.execute_query(
                    """
                    CALL db.index.fulltext.queryNodes($index_name, $query)
                    YIELD node, score
                    WHERE score >= $threshold AND node.id <> $own_page_id
                    RETURN node.id AS page_id
                    LIMIT 3
                    """,
                    {
                        "index_name": settings.fulltext_index_name,
                        "query": section,
                        "threshold": threshold,
                        "own_page_id": ref["page_id"],
                    },
                )
                for row in results:
                    self.db.execute_query(merge_query, {
                        "ref_id": ref["id"],
                        "page_id": row["page_id"],
                    })
                if results:
                    logger.debug(
                        f"Resolved reference '{ref['full_reference']}' → {len(results)} page(s)"
                    )
            except Exception as e:
                logger.warning(f"Could not resolve reference '{ref['full_reference']}': {e}")

    # ------------------------------------------------------------------
    # Table nodes
    # ------------------------------------------------------------------

    def _extract_table_caption(self, markdown: str, placeholder: str) -> str:
        """Find the nearest table caption within 200 chars of the placeholder."""
        pos = markdown.find(f"[{placeholder}]({placeholder})")
        if pos == -1:
            return ""
        window = markdown[max(0, pos - 200): pos + 200]
        match = TABLE_CAPTION_PATTERN.search(window)
        return match.group(0).strip() if match else ""

    def _create_table_nodes(
        self, tables: List[Dict], markdown: str, page_id: str, document_id: str
    ):
        """Persist Table nodes with embeddings and link them to their parent Page."""
        ctx = settings.reference_context_chars
        query = """
            MERGE (t:Table {id: $id})
            SET t.table_id    = $table_id,
                t.page_id     = $page_id,
                t.document_id = $document_id,
                t.content     = $content,
                t.caption     = $caption,
                t.context     = $context,
                t.embedding   = $embedding
            WITH t
            MATCH (p:Page {id: $page_id})
            MERGE (p)-[:HAS_TABLE]->(t)
        """
        for table in tables:
            table_id = table.get("id", "")
            content = table.get("content", "")
            if not table_id or not content:
                continue

            caption = self._extract_table_caption(markdown, table_id)

            placeholder_pos = markdown.find(f"[{table_id}]({table_id})")
            if placeholder_pos != -1:
                context = markdown[
                    max(0, placeholder_pos - ctx): placeholder_pos + ctx
                ]
            else:
                context = ""

            embed_text = f"{caption}\n{context}\n{content[:300]}".strip()
            embedding = self.ollama.generate_embedding(embed_text) if embed_text else []

            self.db.execute_query(query, {
                "id": f"{page_id}_tbl_{table_id}",
                "table_id": table_id,
                "page_id": page_id,
                "document_id": document_id,
                "content": content,
                "caption": caption,
                "context": context,
                "embedding": embedding,
            })
        logger.debug(f"Created {len(tables)} Table node(s) for page {page_id}")

    # ------------------------------------------------------------------
    # Image nodes
    # ------------------------------------------------------------------

    def _extract_image_caption(self, image: Dict, markdown: str) -> str:
        """Return caption from image_annotation or from surrounding markdown text."""
        annotation = image.get("image_annotation")
        if annotation:
            return str(annotation).strip()

        image_id = image.get("id", "")
        if not image_id:
            return ""

        pos = markdown.find(image_id)
        if pos == -1:
            return ""
        window = markdown[max(0, pos - 200): pos + 200]
        match = IMAGE_CAPTION_PATTERN.search(window)
        return match.group(0).strip() if match else ""

    def _create_image_nodes(
        self, images: List[Dict], markdown: str, page_id: str, document_id: str
    ):
        """Persist Image nodes with embeddings and link them to their parent Page."""
        ctx = settings.reference_context_chars
        query = """
            MERGE (i:Image {id: $id})
            SET i.image_id    = $image_id,
                i.page_id     = $page_id,
                i.document_id = $document_id,
                i.caption     = $caption,
                i.context     = $context,
                i.file_path   = $file_path,
                i.embedding   = $embedding
            WITH i
            MATCH (p:Page {id: $page_id})
            MERGE (p)-[:HAS_IMAGE]->(i)
        """
        for image in images:
            image_id = image.get("id", "")
            if not image_id:
                continue

            caption = self._extract_image_caption(image, markdown)

            pos = markdown.find(image_id)
            context = (
                markdown[max(0, pos - ctx): pos + ctx] if pos != -1 else ""
            )

            embed_text = f"{caption}\n{context}".strip()
            if not embed_text:
                continue  # nothing meaningful to embed

            embedding = self.ollama.generate_embedding(embed_text)

            self.db.execute_query(query, {
                "id": f"{page_id}_img_{image_id}",
                "image_id": image_id,
                "page_id": page_id,
                "document_id": document_id,
                "caption": caption,
                "context": context,
                "file_path": image.get("file_path", ""),
                "embedding": embedding,
            })
        logger.debug(f"Created {len(images)} Image node(s) for page {page_id}")
