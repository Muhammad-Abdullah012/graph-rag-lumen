"""Graph Building Module using LLMGraphTransformer"""
import json
import logging
from typing import List, Tuple, Dict, Any
import uuid

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.llms.ollama import Ollama

from backend.app.modules.database import get_neo4j_connection
from backend.app.modules.ollama_client import get_ollama_client
from config.settings import settings

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build knowledge graph from text chunks"""
    
    def __init__(self):
        """Initialize graph builder with LLM"""
        self.db = get_neo4j_connection()
        self.ollama = get_ollama_client()
        
        # Initialize LLM for graph transformation
        self.llm = Ollama(
            base_url=settings.ollama_base_url.rstrip('/'),
            model=settings.ollama_graph_model,
            temperature=0.3,
        )
        
        self.transformer = LLMGraphTransformer(llm=self.llm)
    
    def build_graph(
        self,
        chunks: List[Tuple[str, dict]],
        document_id: str,
        document_name: str,
        document_url: str,
    ):
        """
        Build knowledge graph from text chunks
        
        Args:
            chunks: List of (text, metadata) tuples
            document_id: Unique document identifier
            document_name: Human-readable document name
            document_url: Relative URL to document
        """
        try:
            logger.info(f"Building graph for document: {document_name}")
            
            # Create or update document node
            self._create_document_node(document_id, document_name, document_url)
            
            # Process each chunk
            for idx, (chunk_text, metadata) in enumerate(chunks):
                if not chunk_text.strip():
                    continue
                
                logger.info(f"Processing chunk {idx + 1}/{len(chunks)}")
                
                chunk_id = str(uuid.uuid4())
                
                # Generate embedding
                embedding = self.ollama.generate_embedding(chunk_text)
                
                # Create chunk node
                self._create_chunk_node(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=chunk_text,
                    embedding=embedding,
                    metadata=metadata,
                    page_number=idx  # Approximate page for now
                )
                
                # Transform text to graph
                try:
                    doc = Document(page_content=chunk_text, metadata={"chunk_id": chunk_id})
                    graph_documents = self.transformer.convert_to_graph_documents([doc])
                    
                    # Store graph entities and relationships
                    self._store_graph_entities(
                        graph_documents=graph_documents,
                        chunk_id=chunk_id,
                        document_id=document_id
                    )
                except Exception as e:
                    logger.warning(f"Could not extract entities from chunk {chunk_id}: {str(e)}")
            
            logger.info(f"Successfully built graph for document: {document_name}")
            
        except Exception as e:
            logger.error(f"Error building graph: {str(e)}")
            raise
    
    def _create_document_node(self, doc_id: str, name: str, url: str):
        """Create document node in graph"""
        query = """
            MERGE (d:Document {id: $id})
            SET d.name = $name,
                d.url = $url,
                d.created_at = datetime()
            RETURN d
        """
        
        self.db.execute_query(query, {
            "id": doc_id,
            "name": name,
            "url": url
        })
        
        logger.info(f"Created/updated document node: {doc_id}")
    
    def _create_chunk_node(
        self,
        chunk_id: str,
        document_id: str,
        text: str,
        embedding: list,
        metadata: dict,
        page_number: int
    ):
        """Create document chunk node with embedding"""
        query = """
            MATCH (d:Document {id: $document_id})
            CREATE (c:DocumentChunk {
                id: $chunk_id,
                text: $text,
                embedding: $embedding,
                page_number: $page_number,
                metadata: $metadata
            })
            CREATE (c)-[:FROM_DOCUMENT]->(d)
            RETURN c
        """
        
        self.db.execute_query(query, {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "text": text,
            "embedding": embedding,
            "page_number": page_number,
            "metadata": json.dumps(metadata)
        })
    
    def _store_graph_entities(self, graph_documents, chunk_id: str, document_id: str):
        """Store entities and relationships from graph documents"""
        for graph_doc in graph_documents:
            # Create nodes
            for node in graph_doc.nodes:
                self._create_entity_node(node, chunk_id, document_id)
            
            # Create relationships
            for rel in graph_doc.relationships:
                self._create_relationship(rel, chunk_id)
    
    def _create_entity_node(self, node: Any, chunk_id: str, document_id: str):
        """Create entity node from extracted entity"""
        query = """
            MATCH (c:DocumentChunk {id: $chunk_id})
            MERGE (e {id: $node_id})
            SET e:Entity,
                e.name = $name,
                e.type = $node_type
            CREATE (e)-[:MENTIONED_IN]->(c)
            RETURN e
        """
        
        try:
            self.db.execute_query(query, {
                "chunk_id": chunk_id,
                "node_id": f"{node.id}_{document_id}",
                "name": str(node.id),
                "node_type": node.type if hasattr(node, 'type') else "UNKNOWN"
            })
        except Exception as e:
            logger.debug(f"Could not create entity node: {str(e)}")
    
    def _create_relationship(self, rel: Any, chunk_id: str):
        """Create relationship between entities"""
        query = """
            MATCH (e1 {id: $source_id})
            MATCH (e2 {id: $target_id})
            CREATE (e1)-[r:RELATED]->(e2)
            SET r.type = $rel_type,
                r.found_in_chunk = $chunk_id
            RETURN r
        """
        
        try:
            self.db.execute_query(query, {
                "source_id": str(rel.source.id),
                "target_id": str(rel.target.id),
                "rel_type": rel.type,
                "chunk_id": chunk_id
            })
        except Exception as e:
            logger.debug(f"Could not create relationship: {str(e)}")
