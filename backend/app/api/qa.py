"""Question Answering Routes"""
import logging
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

from backend.app.modules.qa import QASystem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/qa", tags=["qa"])

qa_system = QASystem()


class QuestionRequest(BaseModel):
    """Question request"""
    question: str
    top_k: int = 5


class SourceInfo(BaseModel):
    """Source document information"""
    document_id: str
    document_name: str
    document_url: str
    page_number: int
    relevance_score: float


class AnswerResponse(BaseModel):
    """Question answer response"""
    answer: str
    sources: List[SourceInfo]
    confidence: float


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest) -> AnswerResponse:
    """
    Ask a question and get answer from knowledge graph
    
    Args:
        request: Question request
        
    Returns:
        Answer with sources
    """
    try:
        logger.info(f"Question received: {request.question}")
        
        result = qa_system.answer_question(
            question=request.question,
            top_k=request.top_k
        )
        
        return AnswerResponse(
            answer=result["answer"],
            sources=[SourceInfo(**source) for source in result["sources"]],
            confidence=result["confidence"]
        )
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return AnswerResponse(
            answer=f"Error: {str(e)}",
            sources=[],
            confidence=0.0
        )
