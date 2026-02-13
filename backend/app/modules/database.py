"""Neo4j Database Connection and Management"""
import logging
from typing import Optional, Any, Dict, List

from neo4j import GraphDatabase
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


# Singleton instance
_neo4j_connection = None


def get_neo4j_connection() -> Neo4jConnection:
    """Get or create Neo4j connection"""
    global _neo4j_connection
    if _neo4j_connection is None:
        _neo4j_connection = Neo4jConnection()
    return _neo4j_connection
