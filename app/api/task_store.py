from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

MAX_PROGRESS_EVENTS = 200


class APITaskStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    progress_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                )
                """
            )

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def create_task(self, task_id: str, task_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        now = _now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO api_tasks(task_id, task_type, status, input_json, progress_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (task_id, task_type, "pending", _dumps(payload), _dumps([]), now, now),
            )
        return self.get_task(task_id) or {}

    def mark_running(self, task_id: str) -> None:
        now = _now()
        with self.connect() as conn:
            conn.execute(
                "UPDATE api_tasks SET status=?, started_at=?, updated_at=? WHERE task_id=?",
                ("running", now, now, task_id),
            )

    def add_progress(self, task_id: str, message: str) -> None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status, progress_json FROM api_tasks WHERE task_id=?", (task_id,)).fetchone()
            if not row or row["status"] not in {"pending", "running"}:
                return
            progress = list(_loads(row["progress_json"], []))
            progress.append({"message": message, "created_at": _now()})
            progress = progress[-MAX_PROGRESS_EVENTS:]
            conn.execute(
                "UPDATE api_tasks SET progress_json=?, updated_at=? WHERE task_id=?",
                (_dumps(progress), _now(), task_id),
            )

    def mark_succeeded(self, task_id: str, result: dict[str, Any]) -> None:
        now = _now()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE api_tasks
                SET status=?, result_json=?, error=NULL, updated_at=?, finished_at=?
                WHERE task_id=?
                """,
                ("succeeded", _dumps(result), now, now, task_id),
            )

    def mark_failed(self, task_id: str, error: str) -> None:
        now = _now()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE api_tasks
                SET status=?, error=?, updated_at=?, finished_at=?
                WHERE task_id=?
                """,
                ("failed", error[:4000], now, now, task_id),
            )

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM api_tasks WHERE task_id=?", (task_id,)).fetchone()
        return _row_to_task(row) if row else None


def _row_to_task(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "task_id": row["task_id"],
        "task_type": row["task_type"],
        "status": row["status"],
        "input": _loads(row["input_json"], {}),
        "result": _loads(row["result_json"], None) if row["result_json"] else None,
        "error": row["error"],
        "progress": _loads(row["progress_json"], []),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
    }


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _loads(payload: str, fallback: Any) -> Any:
    try:
        return json.loads(payload)
    except Exception:
        return fallback


def _now() -> str:
    return datetime.utcnow().isoformat()
