"""Neo4j & PostgreSQL Database Connection and Management"""
import logging
from typing import Optional, Any, Dict, List

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from psycopg_pool import AsyncConnectionPool

from config.settings import settings

logger = logging.getLogger(__name__)


# ================================================================== #
#  Neo4j
# ================================================================== #
class Neo4jConnection:
    """Manage Neo4j database connections"""

    def __init__(self):
        """Initialize Neo4j driver"""
        try:
            self.driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
                encrypted=False,
            )
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j successfully")
        except ServiceUnavailable:
            logger.error(f"Could not connect to Neo4j at {settings.neo4j_uri}")
            raise

    def close(self):
        """Close the driver connection"""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return result records."""
        if parameters is None:
            parameters = {}
        if database is None:
            database = settings.neo4j_database
        try:
            with self.driver.session(database=database) as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise


_neo4j_connection: Optional[Neo4jConnection] = None


def get_neo4j_connection() -> Neo4jConnection:
    """Get or create Neo4j connection (singleton)."""
    global _neo4j_connection
    if _neo4j_connection is None:
        _neo4j_connection = Neo4jConnection()
    return _neo4j_connection


# ================================================================== #
#  PostgreSQL (async via psycopg 3)
# ================================================================== #
_pg_pool: Optional[AsyncConnectionPool] = None


async def init_postgres_pool() -> AsyncConnectionPool:
    """Create and open the async connection pool, then bootstrap the schema."""
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool

    logger.info("Initializing PostgreSQL connection pool ...")
    _pg_pool = AsyncConnectionPool(
        conninfo=settings.postgres_dsn,
        min_size=2,
        max_size=10,
        open=False,
    )
    await _pg_pool.open()
    await _bootstrap_pg_schema(_pg_pool)
    logger.info("PostgreSQL pool ready")
    return _pg_pool


async def _bootstrap_pg_schema(pool: AsyncConnectionPool):
    """Create pgvector extension and application tables if they don't exist."""
    async with pool.connection() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")

        # Embeddings table for semantic search
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                id          SERIAL PRIMARY KEY,
                node_type   VARCHAR(50)  NOT NULL,
                node_id     VARCHAR(500) NOT NULL UNIQUE,
                node_name   VARCHAR(500),
                text_content TEXT        NOT NULL,
                embedding   vector({settings.embedding_dimensions}),
                metadata    JSONB        DEFAULT '{{}}'::jsonb,
                created_at  TIMESTAMPTZ  DEFAULT now()
            )
        """)

        # Conversations table (lightweight metadata; messages live in checkpointer)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                title       VARCHAR(500)  DEFAULT 'New Conversation',
                created_at  TIMESTAMPTZ   DEFAULT now(),
                updated_at  TIMESTAMPTZ   DEFAULT now()
            )
        """)

        # Uploaded files table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_files (
                id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                original_filename VARCHAR(500)  NOT NULL,
                stored_filename   VARCHAR(500)  NOT NULL,
                file_size         BIGINT        NOT NULL,
                content_type      VARCHAR(100),
                file_path         VARCHAR(1000) NOT NULL,
                uploaded_at       TIMESTAMPTZ   DEFAULT now()
            )
        """)

        # Indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_type
            ON embeddings(node_type)
        """)
        # IVFFlat index requires rows; create it only if table already has data
        # We'll create/recreate it after embedding sync instead.

        await conn.commit()
    logger.info("PostgreSQL schema bootstrapped")


def get_pg_pool() -> AsyncConnectionPool:
    """Return the initialised pool.  Raises if init_postgres_pool() hasn't been awaited."""
    if _pg_pool is None:
        raise RuntimeError("PostgreSQL pool not initialised – call init_postgres_pool() first")
    return _pg_pool


async def close_postgres_pool():
    """Gracefully close the pool."""
    global _pg_pool
    if _pg_pool is not None:
        await _pg_pool.close()
        _pg_pool = None
        logger.info("PostgreSQL pool closed")
