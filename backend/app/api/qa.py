"""Question Answering Routes - Agent-based Q&A over Eurocode knowledge graph"""
import logging
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

from backend.app.modules.agent import get_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/qa", tags=["qa"])


class QuestionRequest(BaseModel):
    """Question request"""
    question: str


class ToolUsed(BaseModel):
    """Info about a tool the agent called"""
    tool: str
    arguments: Dict[str, Any]


class AnswerResponse(BaseModel):
    """Question answer response"""
    answer: str
    tools_used: List[ToolUsed]
    route: str = ""


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest) -> AnswerResponse:
    """
    Ask a question — the agent queries the Eurocode knowledge graph
    and returns a grounded answer.
    """
    try:
        logger.info(f"Question received: {request.question}")
        agent = get_agent()
        result = await agent.aanswer(request.question)

        return AnswerResponse(
            answer=result["answer"],
            tools_used=[ToolUsed(**t) for t in result.get("tools_used", [])],
            route=result.get("route", ""),
        )

    except Exception as e:
        logger.error(f"Error answering question: {str(e)}", exc_info=True)
        return AnswerResponse(
            answer=f"Error: {str(e)}",
            tools_used=[],
        )
