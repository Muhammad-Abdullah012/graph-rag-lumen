"""Database initialization and management scripts"""
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.modules.database import get_neo4j_connection
from config.logging_config import logger


def init_database():
    """Initialize database with constraints and indexes"""
    logger.info("Initializing database...")
    
    try:
        db = get_neo4j_connection()
        
        # Create constraints
        db.create_constraints()
        
        # Create vector index
        db.create_vector_index()
        
        logger.info("Database initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False


def clear_database():
    """Clear all data from database (USE WITH CAUTION)"""
    if input("Are you sure you want to delete all data? Type 'yes' to confirm: ") != "yes":
        print("Cancelled")
        return
    
    logger.warning("Clearing all data from database...")
    
    try:
        db = get_neo4j_connection()
        
        # Delete all nodes and relationships
        db.execute_query("MATCH (n) DETACH DELETE n")
        
        logger.info("Database cleared successfully")
        
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")


def show_stats():
    """Show database statistics"""
    try:
        db = get_neo4j_connection()
        
        # Count documents
        docs = db.execute_query("MATCH (d:Document) RETURN COUNT(d) as count")
        doc_count = docs[0]['count'] if docs else 0
        
        # Count chunks
        chunks = db.execute_query("MATCH (c:DocumentChunk) RETURN COUNT(c) as count")
        chunk_count = chunks[0]['count'] if chunks else 0
        
        # Count entities
        entities = db.execute_query("MATCH (e:Entity) RETURN COUNT(e) as count")
        entity_count = entities[0]['count'] if entities else 0
        
        print(f"\nDatabase Statistics:")
        print(f"  Documents: {doc_count}")
        print(f"  Chunks: {chunk_count}")
        print(f"  Entities: {entity_count}")
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database management scripts")
    parser.add_argument("command", choices=["init", "clear", "stats"], 
                        help="Command to execute")
    
    args = parser.parse_args()
    
    if args.command == "init":
        init_database()
    elif args.command == "clear":
        clear_database()
    elif args.command == "stats":
        show_stats()
