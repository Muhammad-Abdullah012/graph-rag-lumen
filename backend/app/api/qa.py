"""Question Answering Routes"""
import logging
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.app.modules.qa import QASystem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/qa", tags=["qa"])

qa_system = QASystem()


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


@router.post("/stream")
async def stream_answer(request: QuestionRequest):
    """Stream answer as Server-Sent Events"""
    return StreamingResponse(
        qa_system.answer_question_stream(
            question=request.question,
            top_k=request.top_k,
        ),
        media_type="text/event-stream",
    )
