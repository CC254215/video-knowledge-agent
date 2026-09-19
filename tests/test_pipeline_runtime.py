from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.runtime.pipeline_runtime import dynamic_llm_timeout_seconds, run_stage_with_restarts
from app.runtime.processing_state import ProcessingState


def test_dynamic_llm_timeout_by_video_duration() -> None:
    assert dynamic_llm_timeout_seconds(19 * 60) == 180
    assert dynamic_llm_timeout_seconds(20 * 60) == 180
    assert dynamic_llm_timeout_seconds(30 * 60) == 240
    assert dynamic_llm_timeout_seconds(40 * 60) == 240
    assert dynamic_llm_timeout_seconds(41 * 60) == 360


def test_run_stage_with_restarts_records_problem_and_stops_after_limit(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", pipeline_stage_restart_limit=2)
    state = ProcessingState(tmp_path / "video")
    attempts = {"count": 0}

    def failing_stage() -> str:
        attempts["count"] += 1
        raise RuntimeError("stuck on LLM summary timeout")

    with pytest.raises(RuntimeError, match="stage_restart_exhausted"):
        run_stage_with_restarts("summary", failing_stage, state, settings)

    assert attempts["count"] == 3
    entry = state.payload["stages"]["summary"]
    assert entry["restart_count"] == 3
    assert "stuck on LLM summary timeout" in entry["restarts"][-1]["error"]


def test_run_stage_with_restarts_recovers_before_limit(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", pipeline_stage_restart_limit=2)
    state = ProcessingState(tmp_path / "video")
    attempts = {"count": 0}

    def flaky_stage() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary rate limit")
        return "ok"

    assert run_stage_with_restarts("storyline", flaky_stage, state, settings) == "ok"
    assert attempts["count"] == 2
    assert state.payload["stages"]["storyline"]["restart_count"] == 1


def test_success_clears_stale_stage_error(tmp_path: Path) -> None:
    state = ProcessingState(tmp_path / "video")
    state.start("summary")
    state.fail("summary", "temporary disconnect")

    state.start("summary")
    state.succeed("summary")

    entry = state.payload["stages"]["summary"]
    assert entry["status"] == "succeeded"
    assert "error" not in entry
