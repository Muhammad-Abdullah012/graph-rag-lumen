"""Question Answering Routes - Agent-based Q&A over Eurocode knowledge graph.

This is kept for backward compatibility.  The preferred API is
/api/conversations/{thread_id}/chat (and /chat/stream).
"""
import logging
import uuid

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from backend.app.modules.agent import get_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/qa", tags=["qa"])


class QuestionRequest(BaseModel):
    """Question request"""
    question: str
    thread_id: Optional[str] = None


class ToolUsed(BaseModel):
    """Info about a tool the agent called"""
    tool: str
    arguments: Dict[str, Any]


class AnswerResponse(BaseModel):
    """Question answer response"""
    answer: str
    tools_used: List[ToolUsed]
    thread_id: str


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest) -> AnswerResponse:
    """
    Ask a question.  If thread_id is provided the conversation continues;
    otherwise a throwaway thread is created.
    """
    try:
        logger.info(f"Question received: {request.question}")
        agent = get_agent()
        thread_id = request.thread_id or str(uuid.uuid4())
        result = await agent.aanswer(request.question, thread_id=thread_id)

        return AnswerResponse(
            answer=result["answer"],
            tools_used=[ToolUsed(**t) for t in result.get("tools_used", [])],
            thread_id=thread_id,
        )

    except Exception as e:
        logger.error(f"Error answering question: {str(e)}", exc_info=True)
        return AnswerResponse(
            answer=f"Error: {str(e)}",
            tools_used=[],
            thread_id=request.thread_id or "",
        )
