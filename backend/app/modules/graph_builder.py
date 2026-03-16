"""Graph Building Module — Document and Page nodes"""
import json
import logging
from datetime import datetime, timezone

from backend.app.modules.database import get_neo4j_connection
from backend.app.modules.ollama_client import get_ollama_client

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build a Neo4j graph with Document and Page nodes from OCR page data"""

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
        Create Document + Page nodes and link them.

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
            self._create_page_node(page, document_id)

        self._create_next_page_links(document_id, len(pages))

        logger.info(f"Graph built for document: {document_name}")

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

        # Strip base64 from images (should already be done by pipeline, but be safe)
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
