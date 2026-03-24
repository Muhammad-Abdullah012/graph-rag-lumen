"""Neo4j Database Connection and Management"""
import logging
import time
from typing import Optional, Any, Dict, List

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError

from config.settings import settings

logger = logging.getLogger(__name__)


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
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Database name
            
        Returns:
            List of result records
        """
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
    
    def create_vector_index(self, dimension: int):
        """Create vector similarity index for Page embeddings."""
        try:
            result = self.execute_query(
                "SHOW INDEXES WHERE name = $index_name",
                {"index_name": settings.vector_index_name},
            )
            if result:
                logger.info(f"Vector index '{settings.vector_index_name}' already exists")
                return

            self.execute_query(f"""
                CREATE VECTOR INDEX {settings.vector_index_name}
                FOR (n:Page) ON (n.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {dimension},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """)
            logger.info(f"Created vector index: {settings.vector_index_name}")

        except Exception as e:
            logger.warning(f"Could not create vector index: {str(e)}")
    
    def create_fulltext_index(self):
        """Create fulltext index on Page.markdown for keyword search"""
        try:
            result = self.execute_query(
                "SHOW INDEXES WHERE name = $index_name",
                {"index_name": settings.fulltext_index_name},
            )
            if result:
                logger.info(f"Fulltext index '{settings.fulltext_index_name}' already exists")
                return
            self.execute_query(
                f"CREATE FULLTEXT INDEX {settings.fulltext_index_name} "
                f"FOR (n:Page) ON EACH [n.markdown]"
            )
            logger.info(f"Created fulltext index: {settings.fulltext_index_name}")
        except Exception as e:
            logger.warning(f"Could not create fulltext index: {str(e)}")

    def create_reference_indices(self, dimension: int):
        """Create vector + fulltext indices for Reference nodes."""
        vec_name = settings.reference_vector_index_name
        ft_name = settings.reference_fulltext_index_name
        try:
            existing = {r["name"] for r in self.execute_query("SHOW INDEXES YIELD name")}

            if vec_name not in existing:
                self.execute_query(f"""
                    CREATE VECTOR INDEX {vec_name}
                    FOR (n:Reference) ON (n.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {dimension},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                logger.info(f"Created vector index: {vec_name}")

            if ft_name not in existing:
                self.execute_query(
                    f"CREATE FULLTEXT INDEX {ft_name} "
                    f"FOR (n:Reference) ON EACH [n.full_reference, n.norm, n.section]"
                )
                logger.info(f"Created fulltext index: {ft_name}")
        except Exception as e:
            logger.warning(f"Could not create reference indices: {e}")

    def create_table_indices(self, dimension: int):
        """Create vector + fulltext indices for Table nodes."""
        vec_name = settings.table_vector_index_name
        ft_name = settings.table_fulltext_index_name
        try:
            existing = {r["name"] for r in self.execute_query("SHOW INDEXES YIELD name")}

            if vec_name not in existing:
                self.execute_query(f"""
                    CREATE VECTOR INDEX {vec_name}
                    FOR (n:Table) ON (n.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {dimension},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                logger.info(f"Created vector index: {vec_name}")

            if ft_name not in existing:
                self.execute_query(
                    f"CREATE FULLTEXT INDEX {ft_name} "
                    f"FOR (n:Table) ON EACH [n.content, n.caption]"
                )
                logger.info(f"Created fulltext index: {ft_name}")
        except Exception as e:
            logger.warning(f"Could not create table indices: {e}")

    def create_image_indices(self, dimension: int):
        """Create vector + fulltext indices for Image nodes."""
        vec_name = settings.image_vector_index_name
        ft_name = settings.image_fulltext_index_name
        try:
            existing = {r["name"] for r in self.execute_query("SHOW INDEXES YIELD name")}

            if vec_name not in existing:
                self.execute_query(f"""
                    CREATE VECTOR INDEX {vec_name}
                    FOR (n:Image) ON (n.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {dimension},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                logger.info(f"Created vector index: {vec_name}")

            if ft_name not in existing:
                self.execute_query(
                    f"CREATE FULLTEXT INDEX {ft_name} "
                    f"FOR (n:Image) ON EACH [n.caption]"
                )
                logger.info(f"Created fulltext index: {ft_name}")
        except Exception as e:
            logger.warning(f"Could not create image indices: {e}")

    def create_constraints(self):
        """Create database constraints"""
        constraints = [
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT page_id_unique IF NOT EXISTS FOR (p:Page) REQUIRE p.id IS UNIQUE",
        ]
        
        try:
            for constraint in constraints:
                self.execute_query(constraint)
            logger.info("Database constraints created/verified")
        except Exception as e:
            logger.warning(f"Could not create constraints: {str(e)}")


# Singleton instance
_neo4j_connection = None


def get_neo4j_connection() -> Neo4jConnection:
    """Get or create Neo4j connection, retrying until the database is ready."""
    global _neo4j_connection
    if _neo4j_connection is None:
        max_attempts = 10
        delay = 3  # seconds
        for attempt in range(1, max_attempts + 1):
            try:
                conn = Neo4jConnection()
                conn.create_constraints()
                conn.create_fulltext_index()

                from backend.app.modules.ollama_client import get_ollama_client
                dimension = len(get_ollama_client().generate_embedding("dimension probe"))
                logger.info(f"Detected embedding dimension: {dimension}")

                conn.create_vector_index(dimension)
                conn.create_reference_indices(dimension)
                conn.create_table_indices(dimension)
                conn.create_image_indices(dimension)
                _neo4j_connection = conn
                break
            except (ServiceUnavailable, TransientError) as e:
                if attempt == max_attempts:
                    raise
                logger.warning(
                    f"Neo4j not ready (attempt {attempt}/{max_attempts}): {e} — retrying in {delay}s"
                )
                time.sleep(delay)
    return _neo4j_connection
