from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class EvidenceTraceRecorder:
    def __init__(self, video_id: str, video_dir: str | Path, question: str, trace_id: str | None = None) -> None:
        self.video_id = video_id
        self.video_dir = Path(video_dir)
        self.question = question
        self.trace_id = trace_id or uuid.uuid4().hex
        self.created_at = datetime.utcnow().isoformat()
        self.events: list[dict[str, Any]] = []

    def record(self, event: str, payload: dict[str, Any] | None = None) -> None:
        self.events.append(
            {
                "event": event,
                "timestamp": datetime.utcnow().isoformat(),
                "payload": _jsonable(payload or {}),
            }
        )

    def write(self) -> Path | None:
        try:
            out_dir = self.video_dir / "qa_traces"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{self.trace_id}.json"
            path.write_text(
                json.dumps(
                    {
                        "trace_id": self.trace_id,
                        "video_id": self.video_id,
                        "question": self.question,
                        "created_at": self.created_at,
                        "events": self.events,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return path
        except Exception:
            return None


def evidence_rows_for_trace(rows: list[Any], limit: int = 20) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in rows[:limit]:
        output.append(
            {
                "evidence_id": getattr(item, "evidence_id", None),
                "segment_id": getattr(item, "segment_id", None),
                "evidence_type": getattr(item, "evidence_type", None),
                "start": getattr(item, "start", None),
                "end": getattr(item, "end", None),
                "score": getattr(item, "score", None),
                "text": str(getattr(item, "text", ""))[:500],
            }
        )
    return output


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable({key: getattr(value, key) for key in value.__dataclass_fields__})
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump(mode="json"))
    return str(value)
