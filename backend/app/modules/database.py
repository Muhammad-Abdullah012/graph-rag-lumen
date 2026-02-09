"""Neo4j Database Connection and Management"""
import logging
from typing import Optional, Any, Dict, List

from neo4j import GraphDatabase, Session
from neo4j.exceptions import ServiceUnavailable

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
    
    def create_vector_index(self):
        """Create vector similarity index for embeddings"""
        try:
            # Check if index already exists
            check_query = """
                SHOW INDEXES WHERE name = $index_name
            """
            
            result = self.execute_query(
                check_query,
                {"index_name": settings.vector_index_name}
            )
            
            if result:
                logger.info(f"Vector index '{settings.vector_index_name}' already exists")
                return
            
            # Create the index
            create_query = f"""
                CREATE VECTOR INDEX {settings.vector_index_name}
                FOR (n:DocumentChunk) ON (n.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {settings.vector_dimension},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """
            
            self.execute_query(create_query)
            logger.info(f"Created vector index: {settings.vector_index_name}")
            
        except Exception as e:
            logger.warning(f"Could not create vector index: {str(e)}")
    
    def create_constraints(self):
        """Create database constraints"""
        constraints = [
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:DocumentChunk) REQUIRE c.id IS UNIQUE",
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
    """Get or create Neo4j connection"""
    global _neo4j_connection
    if _neo4j_connection is None:
        _neo4j_connection = Neo4jConnection()
        _neo4j_connection.create_constraints()
        _neo4j_connection.create_vector_index()
    return _neo4j_connection
