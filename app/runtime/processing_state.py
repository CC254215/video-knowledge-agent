from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


TERMINAL_SUCCESS = "succeeded"
RETRYABLE_FAILURE = "failed_retryable"
TERMINAL_FAILURE = "failed_terminal"
RUNNING = "running"
PENDING = "pending"
DEGRADED = "degraded"


class ProcessingState:
    def __init__(self, video_dir: Path) -> None:
        self.video_dir = video_dir
        self.path = video_dir / "processing_state.json"
        self.payload: dict[str, Any] = {"stages": {}, "updated_at": None}
        if self.path.exists():
            try:
                self.payload = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.payload = {"stages": {}, "updated_at": None}

    def stage_status(self, name: str) -> str:
        return str((self.payload.get("stages") or {}).get(name, {}).get("status") or PENDING)

    def is_succeeded(self, name: str, required_paths: list[Path] | None = None) -> bool:
        if self.stage_status(name) != TERMINAL_SUCCESS:
            return False
        return all(path.exists() for path in (required_paths or []))

    def start(self, name: str) -> None:
        self._set(name, RUNNING, data={"started_at": datetime.utcnow().isoformat(), "started_monotonic": time.monotonic()})

    def succeed(self, name: str, artifacts: list[Path] | None = None, extra: dict[str, Any] | None = None) -> None:
        data = dict(extra or {})
        if artifacts:
            data["artifacts"] = [str(path) for path in artifacts]
        data.update(self._duration_data(name))
        self._set(name, TERMINAL_SUCCESS, data=data)

    def degrade(self, name: str, reason: str, artifacts: list[Path] | None = None) -> None:
        data: dict[str, Any] = {"reason": reason}
        if artifacts:
            data["artifacts"] = [str(path) for path in artifacts]
        data.update(self._duration_data(name))
        self._set(name, DEGRADED, data=data)

    def fail(self, name: str, error: Exception | str, retryable: bool = True) -> None:
        self._set(name, RETRYABLE_FAILURE if retryable else TERMINAL_FAILURE, error=str(error).splitlines()[0][:1200], data=self._duration_data(name))

    def record_restart(self, name: str, attempt: int, error: Exception | str) -> None:
        stages = self.payload.setdefault("stages", {})
        entry = dict(stages.get(name) or {})
        restarts = list(entry.get("restarts") or [])
        restarts.append(
            {
                "attempt": attempt,
                "error": str(error).splitlines()[0][:1200],
                "created_at": datetime.utcnow().isoformat(),
            }
        )
        entry["restarts"] = restarts
        entry["restart_count"] = len(restarts)
        stages[name] = entry
        self.payload["updated_at"] = datetime.utcnow().isoformat()
        self.video_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def timings(self) -> dict[str, float]:
        stages = self.payload.get("stages") or {}
        return {
            name: float(value.get("duration_seconds"))
            for name, value in stages.items()
            if isinstance(value, dict) and value.get("duration_seconds") is not None
        }

    def _duration_data(self, name: str) -> dict[str, Any]:
        entry = (self.payload.get("stages") or {}).get(name) or {}
        started = entry.get("started_monotonic")
        if not isinstance(started, (int, float)):
            return {}
        duration = max(0.0, time.monotonic() - float(started))
        return {"finished_at": datetime.utcnow().isoformat(), "duration_seconds": round(duration, 3)}

    def _set(self, name: str, status: str, error: str | None = None, data: dict[str, Any] | None = None) -> None:
        stages = self.payload.setdefault("stages", {})
        entry = dict(stages.get(name) or {})
        entry.update({"status": status, "updated_at": datetime.utcnow().isoformat()})
        if error:
            entry["error"] = error
        if data:
            entry.update(data)
        stages[name] = entry
        self.payload["updated_at"] = datetime.utcnow().isoformat()
        self.video_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
