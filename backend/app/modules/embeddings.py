"""Embedding Service – generate, store, and search embeddings via pgvector + Ollama."""
import json
import logging
from typing import List, Dict, Any, Optional

from langchain_ollama import OllamaEmbeddings

from backend.app.modules.database import get_neo4j_connection, get_pg_pool
from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate embeddings with Ollama and store / query them in pgvector."""

    def __init__(self):
        self.embedder = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embed_model,
        )
        self.dims = settings.embedding_dimensions
        logger.info(
            "EmbeddingService ready  model=%s  dims=%d",
            settings.ollama_embed_model,
            self.dims,
        )

    # ------------------------------------------------------------------ #
    #  Generate
    # ------------------------------------------------------------------ #
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return embedding vectors for a batch of texts."""
        if not texts:
            return []
        return await self.embedder.aembed_documents(texts)

    async def embed_query(self, query: str) -> List[float]:
        """Return the embedding vector for a single query string."""
        return await self.embedder.aembed_query(query)

    # ------------------------------------------------------------------ #
    #  Sync from Neo4j → pgvector
    # ------------------------------------------------------------------ #
    async def sync_graph_embeddings(self) -> Dict[str, int]:
        """
        Pull all relevant nodes from Neo4j, embed their textual
        representation, and upsert into the embeddings table.
        Returns counts per node type.
        """
        db = get_neo4j_connection()
        pool = get_pg_pool()
        stats: Dict[str, int] = {}

        # 1. Symbols
        symbols = db.execute_query(
            """MATCH (sym:Symbol)-[:DEFINED_IN]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
               RETURN sym.id AS id, sym.name AS name, sym.definition AS definition,
                      sym.formula AS formula, sec.name AS section, doc.name AS document"""
        )
        if symbols:
            await self._upsert_nodes(
                pool,
                node_type="symbol",
                rows=symbols,
                id_key="id",
                name_key="name",
                text_fn=lambda r: (
                    f"Symbol: {r['name']}\n"
                    f"Definition: {r.get('definition', '')}\n"
                    f"Formula: {r.get('formula', '')}\n"
                    f"Section: {r.get('section', '')}\n"
                    f"Document: {r.get('document', '')}"
                ),
            )
            stats["symbols"] = len(symbols)

        # 2. Definitions
        definitions = db.execute_query(
            """MATCH (df:Definition)-[:DEFINED_IN]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
               RETURN df.id AS id, df.term AS name, df.definition AS definition,
                      sec.name AS section, doc.name AS document"""
        )
        if definitions:
            await self._upsert_nodes(
                pool,
                node_type="definition",
                rows=definitions,
                id_key="id",
                name_key="name",
                text_fn=lambda r: (
                    f"Definition: {r['name']}\n"
                    f"{r.get('definition', '')}\n"
                    f"Section: {r.get('section', '')}\n"
                    f"Document: {r.get('document', '')}"
                ),
            )
            stats["definitions"] = len(definitions)

        # 3. Abbreviations
        abbreviations = db.execute_query(
            """MATCH (a:Abbreviation)-[:DEFINED_IN]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
               RETURN a.id AS id, a.name AS name, a.definition AS definition,
                      sec.name AS section, doc.name AS document"""
        )
        if abbreviations:
            await self._upsert_nodes(
                pool,
                node_type="abbreviation",
                rows=abbreviations,
                id_key="id",
                name_key="name",
                text_fn=lambda r: (
                    f"Abbreviation: {r['name']}\n"
                    f"Meaning: {r.get('definition', '')}\n"
                    f"Section: {r.get('section', '')}\n"
                    f"Document: {r.get('document', '')}"
                ),
            )
            stats["abbreviations"] = len(abbreviations)

        # 4. Formulas
        formulas = db.execute_query(
            """MATCH (f:Formula)-[:FROM_DOCUMENT]->(doc:Document)
               RETURN f.id AS id, f.name AS name, f.expression AS expression,
                      f.variables AS variables, doc.name AS document"""
        )
        if formulas:
            await self._upsert_nodes(
                pool,
                node_type="formula",
                rows=formulas,
                id_key="id",
                name_key="name",
                text_fn=lambda r: (
                    f"Formula: {r['name']}\n"
                    f"Expression: {r.get('expression', '')}\n"
                    f"Variables: {r.get('variables', '')}\n"
                    f"Document: {r.get('document', '')}"
                ),
            )
            stats["formulas"] = len(formulas)

        # 5. Units
        units = db.execute_query(
            """MATCH (u:Unit)-[:DEFINED_IN]->(sec:Section)-[:BELONGS_TO]->(doc:Document)
               RETURN u.id AS id, u.quantity AS name, u.unit AS unit,
                      sec.name AS section, doc.name AS document"""
        )
        if units:
            await self._upsert_nodes(
                pool,
                node_type="unit",
                rows=units,
                id_key="id",
                name_key="name",
                text_fn=lambda r: (
                    f"Unit: {r['name']}\n"
                    f"Recommended unit: {r.get('unit', '')}\n"
                    f"Section: {r.get('section', '')}\n"
                    f"Document: {r.get('document', '')}"
                ),
            )
            stats["units"] = len(units)

        # Recreate IVFFlat index after bulk load
        await self._ensure_vector_index(pool)

        logger.info(f"Embedding sync complete: {stats}")
        return stats

    # ------------------------------------------------------------------ #
    #  Semantic search
    # ------------------------------------------------------------------ #
    async def search(
        self,
        query: str,
        top_k: int = 10,
        node_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic similarity search.  Returns the closest embeddings to *query*.
        """
        query_vec = await self.embed_query(query)
        pool = get_pg_pool()

        type_filter = ""
        params: Dict[str, Any] = {"vec": str(query_vec), "k": top_k}
        if node_type:
            type_filter = "AND node_type = %(node_type)s"
            params["node_type"] = node_type

        sql = f"""
            SELECT node_type, node_id, node_name, text_content, metadata,
                   1 - (embedding <=> %(vec)s::vector) AS similarity
            FROM   embeddings
            WHERE  embedding IS NOT NULL {type_filter}
            ORDER  BY embedding <=> %(vec)s::vector
            LIMIT  %(k)s
        """

        async with pool.connection() as conn:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()

        results = []
        for row in rows:
            results.append({
                "node_type": row[0],
                "node_id": row[1],
                "node_name": row[2],
                "text": row[3],
                "metadata": row[4] if row[4] else {},
                "similarity": round(float(row[5]), 4) if row[5] else 0,
            })
        return results

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    async def _upsert_nodes(
        self,
        pool,
        node_type: str,
        rows: List[Dict],
        id_key: str,
        name_key: str,
        text_fn,
        batch_size: int = 20,
    ):
        """Embed and upsert a batch of graph nodes."""
        # Determine which nodes are already embedded
        node_ids = [r[id_key] for r in rows]
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT node_id FROM embeddings WHERE node_id = ANY(%(ids)s)",
                {"ids": node_ids},
            )
            existing = {r[0] for r in await cur.fetchall()}

        new_rows = [r for r in rows if r[id_key] not in existing]
        if not new_rows:
            logger.info(f"All {len(rows)} {node_type} nodes already embedded – skipping")
            return

        logger.info(f"Embedding {len(new_rows)} new {node_type} nodes ...")

        for i in range(0, len(new_rows), batch_size):
            batch = new_rows[i : i + batch_size]
            texts = [text_fn(r) for r in batch]
            vectors = await self.embed_texts(texts)

            async with pool.connection() as conn:
                for row, text, vec in zip(batch, texts, vectors):
                    meta = {k: v for k, v in row.items() if k not in (id_key, name_key)}
                    await conn.execute(
                        """INSERT INTO embeddings (node_type, node_id, node_name, text_content, embedding, metadata)
                           VALUES (%(nt)s, %(nid)s, %(nn)s, %(tc)s, %(emb)s::vector, %(meta)s)
                           ON CONFLICT (node_id) DO UPDATE
                              SET text_content = EXCLUDED.text_content,
                                  embedding    = EXCLUDED.embedding,
                                  metadata     = EXCLUDED.metadata""",
                        {
                            "nt": node_type,
                            "nid": row[id_key],
                            "nn": row.get(name_key, ""),
                            "tc": text,
                            "emb": str(vec),
                            "meta": json.dumps(meta, ensure_ascii=False, default=str),
                        },
                    )
                await conn.commit()

    async def _ensure_vector_index(self, pool):
        """Create or replace the IVFFlat index (needs rows to exist)."""
        try:
            async with pool.connection() as conn:
                cur = await conn.execute("SELECT count(*) FROM embeddings WHERE embedding IS NOT NULL")
                cnt = (await cur.fetchone())[0]
                if cnt < 10:
                    logger.info("Too few embeddings for IVFFlat index – skipping")
                    return

                lists = max(1, min(cnt // 10, 100))
                await conn.execute("DROP INDEX IF EXISTS idx_embeddings_vector")
                await conn.execute(
                    f"""CREATE INDEX idx_embeddings_vector
                        ON embeddings USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = {lists})"""
                )
                await conn.commit()
                logger.info(f"IVFFlat index created with {lists} lists over {cnt} embeddings")
        except Exception as e:
            logger.warning(f"Could not create vector index: {e}")


# ------------------------------------------------------------------ #
#  Singleton
# ------------------------------------------------------------------ #
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
