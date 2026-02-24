"""Question Answering Routes - Agent-based Q&A over Eurocode knowledge graph"""
import json
import logging
from typing import Any, AsyncGenerator, Dict, List

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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
    and returns a grounded answer (non-streaming).
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


@router.post("/stream")
async def stream_question(request: QuestionRequest) -> StreamingResponse:
    """
    Ask a question and receive a streaming Server-Sent Events (SSE) response.

    Event types emitted:
      data: {"type": "status",  "step": "...", "message": "..."}
      data: {"type": "token",   "content": "..."}
      data: {"type": "done",    "answer": "...", "tools_used": [...], "route": "..."}

    The frontend should accumulate "token" events for immediate display, then
    replace the assembled text with the "done" answer (which has been
    post-processed, e.g. LaTeX delimiters normalised).
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            agent = get_agent()
            async for event in agent.astream_answer(request.question):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error("Streaming QA error: %s", e, exc_info=True)
            error_event = {"type": "done", "answer": f"Fehler: {e}", "tools_used": [], "route": ""}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection": "keep-alive",
        },
    )
