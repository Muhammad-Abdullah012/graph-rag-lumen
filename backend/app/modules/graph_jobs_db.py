"""PostgreSQL-backed job status store for long-running graph operations."""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool

from config.settings import settings

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS graph_jobs (
    id          SERIAL PRIMARY KEY,
    job_name    VARCHAR(100) NOT NULL,
    status      VARCHAR(50)  NOT NULL DEFAULT 'running',
    started_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    result      JSONB,
    error       TEXT
);
"""

_pool: Optional[SimpleConnectionPool] = None


def _get_pool() -> SimpleConnectionPool:
    global _pool
    if _pool is not None:
        return _pool

    _pool = SimpleConnectionPool(
        minconn=1,
        maxconn=3,
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        dbname=settings.postgres_db,
        connect_timeout=10,
    )
    conn = _pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(_DDL)
        conn.commit()
        logger.info("Graph jobs table ready (Postgres)")
    finally:
        _pool.putconn(conn)

    return _pool


def _exec(sql: str, params: tuple = ()) -> Optional[Any]:
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            conn.commit()
            try:
                return cur.fetchone()
            except Exception:
                return None
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def _row_to_dict(row) -> Dict[str, Any]:
    if row is None:
        return {}
    d = dict(row)
    for key in ("started_at", "finished_at"):
        if isinstance(d.get(key), datetime):
            d[key] = d[key].isoformat()
    if isinstance(d.get("result"), str):
        try:
            d["result"] = json.loads(d["result"])
        except Exception:
            pass
    return d


# ── Public API ────────────────────────────────────────────────────────────────

def get_running_job() -> Optional[Dict[str, Any]]:
    """Return the currently running job row, or None."""
    try:
        pool = _get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM graph_jobs WHERE status = 'running' ORDER BY started_at DESC LIMIT 1"
                )
                row = cur.fetchone()
                return _row_to_dict(row) if row else None
        finally:
            pool.putconn(conn)
    except Exception as e:
        logger.error("Failed to get running job: %s", e)
        return None


def start_job(name: str) -> int:
    """Insert a new running job row and return its id."""
    row = _exec(
        "INSERT INTO graph_jobs (job_name, status) VALUES (%s, 'running') RETURNING id",
        (name,),
    )
    return row["id"]


def finish_job(job_id: int, result: Any) -> None:
    """Mark job as succeeded with a result payload."""
    _exec(
        """UPDATE graph_jobs
           SET status = 'success', finished_at = NOW(), result = %s
           WHERE id = %s""",
        (json.dumps(result, default=str), job_id),
    )


def fail_job(job_id: int, error: str) -> None:
    """Mark job as failed with an error message."""
    _exec(
        """UPDATE graph_jobs
           SET status = 'failed', finished_at = NOW(), error = %s
           WHERE id = %s""",
        (error, job_id),
    )


def get_latest_job() -> Optional[Dict[str, Any]]:
    """Return the most recent job row (running or finished)."""
    try:
        pool = _get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM graph_jobs ORDER BY started_at DESC LIMIT 1")
                row = cur.fetchone()
                return _row_to_dict(row) if row else None
        finally:
            pool.putconn(conn)
    except Exception as e:
        logger.error("Failed to get latest job: %s", e)
        return None


def get_job_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Return recent jobs ordered by start time descending."""
    try:
        pool = _get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM graph_jobs ORDER BY started_at DESC LIMIT %s",
                    (limit,),
                )
                return [_row_to_dict(row) for row in cur.fetchall()]
        finally:
            pool.putconn(conn)
    except Exception as e:
        logger.error("Failed to get job history: %s", e)
        return []
