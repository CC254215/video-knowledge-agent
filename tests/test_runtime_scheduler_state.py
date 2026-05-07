from pathlib import Path

import pytest

from app.config import Settings
from app.runtime.processing_state import ProcessingState
from app.runtime.scheduler import ApiScheduler


def test_api_scheduler_retries_retryable_failure():
    settings = Settings(
        _env_file=None,
        api_max_retries=2,
        api_retry_base_delay_seconds=0.01,
        api_rate_limit_cooldown_seconds=0.01,
        llm_min_interval_seconds=0,
    )
    scheduler = ApiScheduler(settings)
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise RuntimeError("429 rate limit")
        return "ok"

    assert scheduler.call("llm", flaky) == "ok"
    assert calls["count"] == 2


def test_processing_state_records_and_resumes_stage(tmp_path: Path):
    artifact = tmp_path / "summary.json"
    artifact.write_text("{}", encoding="utf-8")
    state = ProcessingState(tmp_path)
    state.start("summary")
    state.succeed("summary", [artifact])

    reloaded = ProcessingState(tmp_path)
    assert reloaded.is_succeeded("summary", [artifact])

    artifact.unlink()
    assert not reloaded.is_succeeded("summary", [artifact])


def test_api_scheduler_stops_after_retry_budget():
    settings = Settings(
        _env_file=None,
        api_max_retries=1,
        api_retry_base_delay_seconds=0.01,
        api_rate_limit_cooldown_seconds=0.01,
        llm_min_interval_seconds=0,
    )
    scheduler = ApiScheduler(settings)

    with pytest.raises(RuntimeError):
        scheduler.call("llm", lambda: (_ for _ in ()).throw(RuntimeError("429 rate limit")))


def test_api_scheduler_applies_rate_limit_cooldown():
    settings = Settings(
        _env_file=None,
        api_max_retries=1,
        api_retry_base_delay_seconds=0.01,
        api_rate_limit_cooldown_seconds=0.02,
        llm_min_interval_seconds=0,
    )
    scheduler = ApiScheduler(settings)
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("HTTP 429 Too Many Requests")
        return "ok"

    assert scheduler.call("llm", flaky) == "ok"
    assert calls["count"] == 2
