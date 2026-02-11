"""Question Answering Module with Source Attribution"""
import logging
from typing import List, Dict, Any, Tuple

from backend.app.modules.database import get_neo4j_connection
from backend.app.modules.ollama_client import get_ollama_client
from config.settings import settings

logger = logging.getLogger(__name__)


class QASystem:
    """Answer questions using knowledge graph and retrieval"""
    
    def __init__(self):
        """Initialize QA system"""
        self.db = get_neo4j_connection()
        self.ollama = get_ollama_client()
    
    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer a question using knowledge graph
        
        Args:
            question: User question
            top_k: Number of top chunks to retrieve
            
        Returns:
            Dict with answer and sources
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Generate embedding for question
            question_embedding = self.ollama.generate_embedding(question)
            
            # Retrieve relevant chunks using vector similarity
            relevant_chunks = self._retrieve_relevant_chunks(
                question,
                question_embedding,
                top_k=top_k
            )
            
            if not relevant_chunks:
                logger.warning("No relevant chunks found for question")
                return {
                    "answer": "I could not find relevant information to answer this question.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Build context from chunks
            context = self._build_context(relevant_chunks)
            
            # Generate answer using LLM
            answer = self._generate_answer(question, context)
            
            # Extract sources
            sources = self._extract_sources(relevant_chunks)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": self._calculate_confidence(relevant_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            raise
    
    def _retrieve_relevant_chunks(
        self,
        question: str,
        question_embedding: list,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Retrieve most similar chunks using vector index"""
        query = f"""
            CALL db.index.vector.queryNodes(
                '{settings.vector_index_name}',
                $top_k,
                $embedding
            ) YIELD node, score
            MATCH (node)-[:FROM_DOCUMENT]->(doc:Document)
            RETURN 
                node.id as chunk_id,
                node.text as text,
                node.page_number as page_number,
                doc.id as document_id,
                doc.name as document_name,
                doc.url as document_url,
                score
            ORDER BY score DESC
        """
        
        try:
            results = self.db.execute_query(query, {
                "top_k": top_k,
                "embedding": question_embedding
            })
            
            logger.info(f"Retrieved {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.warning(f"Vector search failed: {str(e)}. Falling back to text search.")
            return self._retrieve_chunks_by_keyword(question, top_k)
    
    def _retrieve_chunks_by_keyword(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback: retrieve chunks by keyword matching"""
        words = [word.lower() for word in query_text.strip().split()[:5] if word]
        
        if not words:
            return []
        
        cypher_query = """
            MATCH (node:DocumentChunk)-[:FROM_DOCUMENT]->(doc:Document)
            WHERE ANY(word IN $words WHERE toLower(node.text) CONTAINS word)
            RETURN 
                node.id AS chunk_id,
                node.text AS text,
                node.page_number AS page_number,
                doc.id AS document_id,
                doc.name AS document_name,
                doc.url AS document_url,
                0.5 AS score
            LIMIT $top_k
        """
        
        try:
            return self.db.execute_query(
                cypher_query,
                parameters={"words": words, "top_k": top_k}
            )
        except Exception as e:
            logger.warning(f"Keyword search failed: {str(e)}")
            return []
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context from retrieved chunks"""
        context_parts = []
        
        for chunk in chunks[:3]:  # Use top 3 chunks
            text = chunk.get("text", "")
            doc_name = chunk.get("document_name", "Unknown")
            context_parts.append(f"[From {doc_name}]\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM"""
        prompt = f"""You are a helpful assistant answering questions based on provided documents.

Question: {question}

Context:
{context}

Based on the context provided above, answer the question concisely and accurately. If the context doesn't contain relevant information, say so.

Answer:"""
        
        try:
            answer = self.ollama.generate_text(
                prompt=prompt,
                model=settings.ollama_llm_model,
                temperature=0.5,
            )
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I encountered an error while generating the answer."
    
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from chunks"""
        sources = []
        seen_docs = set()
        
        for chunk in chunks:
            doc_id = chunk.get("document_id")
            
            if doc_id not in seen_docs:
                sources.append({
                    "document_id": doc_id,
                    "document_name": chunk.get("document_name", "Unknown"),
                    "document_url": chunk.get("document_url", ""),
                    "page_number": chunk.get("page_number", 0),
                    "relevance_score": float(chunk.get("score", 0))
                })
                seen_docs.add(doc_id)
        
        return sources
    
    def _calculate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieval scores"""
        if not chunks:
            return 0.0
        
        scores = [float(chunk.get("score", 0)) for chunk in chunks[:3]]
        return sum(scores) / len(scores) if scores else 0.0
