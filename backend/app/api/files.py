"""File Upload Routes – upload, list, and delete files (storage only, no processing)."""
import logging
import os
import uuid
from pathlib import Path
from typing import List

import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from backend.app.modules.database import get_pg_pool
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"])

# Ensure upload directory exists
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
#  Schemas
# ------------------------------------------------------------------ #
class FileOut(BaseModel):
    id: str
    original_filename: str
    file_size: int
    content_type: str | None = None
    uploaded_at: str


class FileUploadResponse(BaseModel):
    files: List[FileOut]
    message: str


# ------------------------------------------------------------------ #
#  Upload (one or many files)
# ------------------------------------------------------------------ #
@router.post("/upload", response_model=FileUploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload one or more files.  Files are stored on disk; metadata is
    recorded in PostgreSQL.  No processing is triggered.
    """
    pool = get_pg_pool()
    uploaded: List[FileOut] = []

    for file in files:
        if file.size and file.size > settings.max_upload_size:
            raise HTTPException(
                status_code=413,
                detail=f"File '{file.filename}' exceeds the {settings.max_upload_size // (1024*1024)}MB limit",
            )

        file_id = str(uuid.uuid4())
        ext = Path(file.filename or "file").suffix
        stored_name = f"{file_id}{ext}"
        file_path = os.path.join(settings.upload_dir, stored_name)

        # Stream to disk
        content = await file.read()
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        file_size = len(content)

        # Persist metadata
        async with pool.connection() as conn:
            await conn.execute(
                """INSERT INTO uploaded_files
                       (id, original_filename, stored_filename, file_size, content_type, file_path)
                   VALUES (%(id)s, %(orig)s, %(stored)s, %(size)s, %(ct)s, %(path)s)""",
                {
                    "id": file_id,
                    "orig": file.filename or "unknown",
                    "stored": stored_name,
                    "size": file_size,
                    "ct": file.content_type,
                    "path": file_path,
                },
            )
            await conn.commit()

            cur = await conn.execute(
                "SELECT id, original_filename, file_size, content_type, uploaded_at "
                "FROM uploaded_files WHERE id = %(id)s",
                {"id": file_id},
            )
            row = await cur.fetchone()

        uploaded.append(
            FileOut(
                id=str(row[0]),
                original_filename=row[1],
                file_size=row[2],
                content_type=row[3],
                uploaded_at=row[4].isoformat(),
            )
        )
        logger.info("Stored file %s → %s (%d bytes)", file.filename, stored_name, file_size)

    return FileUploadResponse(
        files=uploaded,
        message=f"{len(uploaded)} file(s) uploaded successfully",
    )


# ------------------------------------------------------------------ #
#  List
# ------------------------------------------------------------------ #
@router.get("/", response_model=List[FileOut])
async def list_files():
    """Return all uploaded files, newest first."""
    pool = get_pg_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id, original_filename, file_size, content_type, uploaded_at "
            "FROM uploaded_files ORDER BY uploaded_at DESC"
        )
        rows = await cur.fetchall()

    return [
        FileOut(
            id=str(r[0]),
            original_filename=r[1],
            file_size=r[2],
            content_type=r[3],
            uploaded_at=r[4].isoformat(),
        )
        for r in rows
    ]


# ------------------------------------------------------------------ #
#  Delete
# ------------------------------------------------------------------ #
@router.delete("/{file_id}", status_code=204)
async def delete_file(file_id: str):
    """Delete an uploaded file (disk + database)."""
    pool = get_pg_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT file_path FROM uploaded_files WHERE id = %(id)s",
            {"id": file_id},
        )
        row = await cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="File not found")

        file_path = row[0]

        await conn.execute(
            "DELETE FROM uploaded_files WHERE id = %(id)s",
            {"id": file_id},
        )
        await conn.commit()

    # Remove from disk (best-effort)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError as e:
        logger.warning("Could not delete file from disk: %s", e)
