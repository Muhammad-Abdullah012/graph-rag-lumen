"""Conversation Management Routes – CRUD + streaming chat."""
import json
import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.app.modules.agent import get_agent
from backend.app.modules.database import get_pg_pool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


# ------------------------------------------------------------------ #
#  Schemas
# ------------------------------------------------------------------ #
class ConversationCreate(BaseModel):
    title: Optional[str] = "New Conversation"


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class MessageRequest(BaseModel):
    message: str


class MessageOut(BaseModel):
    role: str
    content: str
    tool_calls: Optional[list] = None


class AnswerResponse(BaseModel):
    answer: str
    tools_used: list = []


# ------------------------------------------------------------------ #
#  Conversation CRUD
# ------------------------------------------------------------------ #
@router.post("/", response_model=ConversationOut, status_code=201)
async def create_conversation(body: ConversationCreate):
    """Create a new conversation thread."""
    pool = get_pg_pool()
    conv_id = str(uuid.uuid4())
    async with pool.connection() as conn:
        await conn.execute(
            """INSERT INTO conversations (id, title)
               VALUES (%(id)s, %(title)s)""",
            {"id": conv_id, "title": body.title or "New Conversation"},
        )
        await conn.commit()
        cur = await conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = %(id)s",
            {"id": conv_id},
        )
        row = await cur.fetchone()

    return ConversationOut(
        id=str(row[0]),
        title=row[1],
        created_at=row[2].isoformat(),
        updated_at=row[3].isoformat(),
    )


@router.get("/", response_model=List[ConversationOut])
async def list_conversations():
    """List all conversations, newest first."""
    pool = get_pg_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        )
        rows = await cur.fetchall()

    return [
        ConversationOut(
            id=str(r[0]), title=r[1],
            created_at=r[2].isoformat(), updated_at=r[3].isoformat(),
        )
        for r in rows
    ]


@router.get("/{conversation_id}", response_model=ConversationOut)
async def get_conversation(conversation_id: str):
    """Get conversation metadata."""
    pool = get_pg_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = %(id)s",
            {"id": conversation_id},
        )
        row = await cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationOut(
        id=str(row[0]), title=row[1],
        created_at=row[2].isoformat(), updated_at=row[3].isoformat(),
    )


@router.patch("/{conversation_id}", response_model=ConversationOut)
async def update_conversation(conversation_id: str, body: ConversationCreate):
    """Rename a conversation."""
    pool = get_pg_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """UPDATE conversations SET title = %(title)s, updated_at = now()
               WHERE id = %(id)s""",
            {"id": conversation_id, "title": body.title},
        )
        await conn.commit()
        cur = await conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = %(id)s",
            {"id": conversation_id},
        )
        row = await cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationOut(
        id=str(row[0]), title=row[1],
        created_at=row[2].isoformat(), updated_at=row[3].isoformat(),
    )


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str):
    """Delete a conversation and its checkpointer data."""
    pool = get_pg_pool()
    async with pool.connection() as conn:
        # Delete from our metadata table
        await conn.execute(
            "DELETE FROM conversations WHERE id = %(id)s",
            {"id": conversation_id},
        )
        # Also clean up checkpointer tables (best-effort)
        for tbl in ("checkpoint_writes", "checkpoints"):
            try:
                await conn.execute(
                    f"DELETE FROM {tbl} WHERE thread_id = %(tid)s",
                    {"tid": conversation_id},
                )
            except Exception:
                pass
        await conn.commit()


# ------------------------------------------------------------------ #
#  Messages (history)
# ------------------------------------------------------------------ #
@router.get("/{conversation_id}/messages", response_model=List[MessageOut])
async def get_messages(conversation_id: str):
    """Retrieve the full message history for a conversation."""
    agent = get_agent()
    messages = await agent.get_thread_messages(conversation_id)
    return [MessageOut(**m) for m in messages]


# ------------------------------------------------------------------ #
#  Chat – non-streaming
# ------------------------------------------------------------------ #
@router.post("/{conversation_id}/chat", response_model=AnswerResponse)
async def chat(conversation_id: str, body: MessageRequest):
    """Send a message and get a complete (non-streamed) response."""
    agent = get_agent()

    # Touch updated_at
    pool = get_pg_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE conversations SET updated_at = now() WHERE id = %(id)s",
            {"id": conversation_id},
        )
        await conn.commit()

    result = await agent.aanswer(body.message, thread_id=conversation_id)
    return AnswerResponse(
        answer=result["answer"],
        tools_used=result.get("tools_used", []),
    )


# ------------------------------------------------------------------ #
#  Chat – streaming (SSE)
# ------------------------------------------------------------------ #
@router.post("/{conversation_id}/chat/stream")
async def chat_stream(conversation_id: str, body: MessageRequest):
    """Send a message and receive the response as a Server-Sent Events stream."""
    agent = get_agent()

    # Touch updated_at
    pool = get_pg_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE conversations SET updated_at = now() WHERE id = %(id)s",
            {"id": conversation_id},
        )
        await conn.commit()

    return StreamingResponse(
        agent.astream_answer(body.message, thread_id=conversation_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
