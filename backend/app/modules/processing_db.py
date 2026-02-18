"""PostgreSQL-backed document processing status store."""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Schema
# ---------------------------------------------------------------------------
_DDL = """
CREATE TABLE IF NOT EXISTS document_processing_status (
    id              SERIAL PRIMARY KEY,
    filename        VARCHAR(512) UNIQUE NOT NULL,
    status          VARCHAR(50)  NOT NULL DEFAULT 'pending',
    step            VARCHAR(100),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    error           TEXT,
    stats           JSONB,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
"""

# ---------------------------------------------------------------------------
#  Connection pool (lazy)
# ---------------------------------------------------------------------------
_pool: Optional[SimpleConnectionPool] = None


def _get_pool() -> SimpleConnectionPool:
    global _pool
    if _pool is not None:
        return _pool

    try:
        _pool = SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            dbname=settings.postgres_db,
            connect_timeout=10,
        )
        # Ensure table exists on first connection
        conn = _pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(_DDL)
            conn.commit()
            logger.info("Processing status table ready (Postgres)")
        finally:
            _pool.putconn(conn)
    except Exception as e:
        logger.error("Could not connect to Postgres for status tracking: %s", e)
        _pool = None
        raise

    return _pool


# ---------------------------------------------------------------------------
#  Public helpers
# ---------------------------------------------------------------------------

def upsert_status(
    filename: str,
    status: str,
    step: str,
    error: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> None:
    """Insert or update the processing status row for *filename*."""
    try:
        pool = _get_pool()
    except Exception:
        logger.warning("Postgres unavailable – skipping status upsert for %s", filename)
        return

    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_processing_status
                    (filename, status, step, started_at, completed_at, error, stats)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (filename) DO UPDATE SET
                    status       = EXCLUDED.status,
                    step         = EXCLUDED.step,
                    started_at   = COALESCE(EXCLUDED.started_at,
                                            document_processing_status.started_at),
                    completed_at = EXCLUDED.completed_at,
                    error        = EXCLUDED.error,
                    stats        = EXCLUDED.stats,
                    updated_at   = NOW()
                """,
                (
                    filename,
                    status,
                    step,
                    started_at,
                    completed_at,
                    error,
                    json.dumps(stats, default=str) if stats else None,
                ),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error("Failed to upsert processing status for %s: %s", filename, e)
    finally:
        pool.putconn(conn)


def get_status(filename: str) -> Optional[Dict[str, Any]]:
    """Return the status row for *filename*, or ``None`` if not found."""
    try:
        pool = _get_pool()
    except Exception:
        return None

    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM document_processing_status WHERE filename = %s",
                (filename,),
            )
            row = cur.fetchone()
            return _row_to_dict(row) if row else None
    except Exception as e:
        logger.error("Failed to get processing status for %s: %s", filename, e)
        return None
    finally:
        pool.putconn(conn)


def get_all_statuses() -> List[Dict[str, Any]]:
    """Return all status rows ordered by most-recently-updated first."""
    try:
        pool = _get_pool()
    except Exception:
        return []

    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM document_processing_status ORDER BY updated_at DESC"
            )
            return [_row_to_dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error("Failed to get all processing statuses: %s", e)
        return []
    finally:
        pool.putconn(conn)


# ---------------------------------------------------------------------------
#  Internal
# ---------------------------------------------------------------------------

def _row_to_dict(row) -> Dict[str, Any]:
    d = dict(row)
    # Convert datetime objects to ISO strings for JSON serialisation
    for key in ("started_at", "completed_at", "created_at", "updated_at"):
        if isinstance(d.get(key), datetime):
            d[key] = d[key].isoformat()
    # stats is already a dict when coming from JSONB, but may be a string
    if isinstance(d.get("stats"), str):
        try:
            d["stats"] = json.loads(d["stats"])
        except Exception:
            pass
    return d
