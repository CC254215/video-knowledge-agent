from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Callable, TypeVar

from app.config import Settings
from app.models import VideoMetadata
from app.runtime.processing_state import ProcessingState
from app.runtime.scheduler import is_rate_limit_error

logger = logging.getLogger(__name__)
T = TypeVar("T")


def dynamic_llm_timeout_seconds(duration_seconds: float | None) -> float:
    """Return max per-request LLM response time based on video duration."""
    duration = float(duration_seconds or 0.0)
    if duration <= 20 * 60:
        return 180.0
    if duration <= 40 * 60:
        return 240.0
    return 360.0


@contextmanager
def temporary_request_timeout(settings: Settings, timeout_seconds: float):
    original = settings.request_timeout_seconds
    settings.request_timeout_seconds = timeout_seconds
    try:
        yield
    finally:
        settings.request_timeout_seconds = original


def apply_dynamic_llm_timeout(settings: Settings, metadata: VideoMetadata) -> float:
    timeout = dynamic_llm_timeout_seconds(metadata.duration)
    settings.request_timeout_seconds = timeout
    logger.info(
        "Dynamic LLM timeout applied: video_duration=%.1fs timeout=%.1fs",
        float(metadata.duration or 0.0),
        timeout,
    )
    return timeout


def run_stage_with_restarts(
    stage_name: str,
    func: Callable[[], T],
    state: ProcessingState,
    settings: Settings,
) -> T:
    """Run a pipeline stage with bounded restart attempts.

    The stage is retried for transient stalls/failures up to
    PIPELINE_STAGE_RESTART_LIMIT. Each failed attempt is written into
    processing_state.json with the error content for audit.
    """
    attempts = max(0, int(settings.pipeline_stage_restart_limit)) + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            state.record_restart(stage_name, attempt, exc)
            logger.warning(
                "Pipeline stage failed; restart attempt recorded: stage=%s attempt=%s/%s error=%s",
                stage_name,
                attempt,
                attempts,
                str(exc).splitlines()[0],
            )
            if attempt >= attempts:
                break
            if is_rate_limit_error(exc):
                wait_for = max(float(settings.api_rate_limit_cooldown_seconds), 1.5 * attempt)
                logger.warning("Rate limit detected; waiting %.1fs before restarting stage=%s", wait_for, stage_name)
                time.sleep(wait_for)
            else:
                time.sleep(min(5.0, 1.5 * attempt))
    assert last_error is not None
    raise RuntimeError(
        f"stage_restart_exhausted: stage={stage_name}, max_restarts={settings.pipeline_stage_restart_limit}, "
        f"last_error={str(last_error).splitlines()[0]}"
    ) from last_error
