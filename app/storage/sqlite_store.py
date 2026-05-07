from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from app.models import ConversationTurn, QAAnswer, Storyline, SummaryReport, TranscriptSegment, VideoMetadata


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists videos (
                  video_id text primary key,
                  metadata_json text not null,
                  created_at text not null
                );
                create table if not exists transcript_segments (
                  video_id text not null,
                  segment_id text not null,
                  segment_json text not null,
                  primary key (video_id, segment_id)
                );
                create table if not exists summaries (
                  video_id text primary key,
                  summary_json text not null,
                  created_at text not null
                );
                create table if not exists storylines (
                  video_id text primary key,
                  storyline_json text not null,
                  created_at text not null
                );
                create table if not exists qa_history (
                  id integer primary key autoincrement,
                  video_id text not null,
                  qa_json text not null,
                  created_at text not null
                );
                create table if not exists processing_runs (
                  id integer primary key autoincrement,
                  video_id text,
                  started_at text not null,
                  finished_at text,
                  status text not null,
                  errors text,
                  model_used text,
                  modality_mode text
                );
                """
            )

    def save_video(self, metadata: VideoMetadata) -> None:
        with self.connect() as conn:
            conn.execute(
                "insert or replace into videos values (?, ?, ?)",
                (metadata.video_id, metadata.model_dump_json(), datetime.utcnow().isoformat()),
            )

    def save_segments(self, video_id: str, segments: list[TranscriptSegment]) -> None:
        with self.connect() as conn:
            conn.executemany(
                "insert or replace into transcript_segments values (?, ?, ?)",
                [(video_id, segment.segment_id, segment.model_dump_json()) for segment in segments],
            )

    def save_summary(self, summary: SummaryReport) -> None:
        with self.connect() as conn:
            conn.execute(
                "insert or replace into summaries values (?, ?, ?)",
                (summary.video_id, summary.model_dump_json(), datetime.utcnow().isoformat()),
            )

    def save_storyline(self, storyline: Storyline) -> None:
        with self.connect() as conn:
            conn.execute(
                "insert or replace into storylines values (?, ?, ?)",
                (storyline.video_id, storyline.model_dump_json(), datetime.utcnow().isoformat()),
            )

    def save_qa(self, video_id: str, answer: QAAnswer) -> None:
        with self.connect() as conn:
            conn.execute(
                "insert into qa_history (video_id, qa_json, created_at) values (?, ?, ?)",
                (video_id, answer.model_dump_json(), datetime.utcnow().isoformat()),
            )

    def save_conversation_turn(self, turn: ConversationTurn) -> None:
        with self.connect() as conn:
            conn.execute(
                "insert into qa_history (video_id, qa_json, created_at) values (?, ?, ?)",
                (turn.video_id, turn.model_dump_json(), datetime.utcnow().isoformat()),
            )

    def start_run(self, video_id: str | None = None, model_used: str | None = None) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                "insert into processing_runs (video_id, started_at, status, model_used) values (?, ?, ?, ?)",
                (video_id, datetime.utcnow().isoformat(), "running", model_used),
            )
            return int(cursor.lastrowid)

    def finish_run(self, run_id: int, status: str, errors: str | None = None, modality_mode: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                "update processing_runs set finished_at=?, status=?, errors=?, modality_mode=? where id=?",
                (datetime.utcnow().isoformat(), status, errors, modality_mode, run_id),
            )

    def raw_counts(self) -> dict[str, int]:
        tables = ["videos", "transcript_segments", "summaries", "storylines", "qa_history", "processing_runs"]
        with self.connect() as conn:
            return {table: int(conn.execute(f"select count(*) from {table}").fetchone()[0]) for table in tables}


def dumps_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
