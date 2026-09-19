from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from app.config import Settings, get_settings

T = TypeVar("T")


@dataclass(frozen=True)
class ResourcePolicy:
    max_concurrency: int
    min_interval_seconds: float


class RateLimitedResource:
    def __init__(self, policy: ResourcePolicy) -> None:
        self.policy = policy
        self._semaphore = threading.BoundedSemaphore(max(1, policy.max_concurrency))
        self._lock = threading.Lock()
        self._last_started = 0.0
        self._cooldown_until = 0.0

    def run(self, fn: Callable[[], T]) -> T:
        with self._semaphore:
            with self._lock:
                now = time.monotonic()
                cooldown_wait = self._cooldown_until - now
                if cooldown_wait > 0:
                    time.sleep(cooldown_wait)
                elapsed = time.monotonic() - self._last_started
                wait_for = self.policy.min_interval_seconds - elapsed
                if wait_for > 0:
                    time.sleep(wait_for)
                self._last_started = time.monotonic()
            return fn()

    def cooldown(self, seconds: float) -> None:
        if seconds <= 0:
            return
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + seconds)


class ApiScheduler:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._resources: dict[str, RateLimitedResource] = {
            "llm": RateLimitedResource(ResourcePolicy(self.settings.llm_max_concurrency, self.settings.llm_min_interval_seconds)),
            "vlm": RateLimitedResource(ResourcePolicy(self.settings.vlm_caption_concurrency, self.settings.vlm_min_interval_seconds)),
            "embedding": RateLimitedResource(ResourcePolicy(self.settings.embedding_max_concurrency, self.settings.embedding_min_interval_seconds)),
            "ytdlp": RateLimitedResource(ResourcePolicy(self.settings.ytdlp_max_concurrency, self.settings.ytdlp_min_interval_seconds)),
            "asr": RateLimitedResource(ResourcePolicy(self.settings.asr_max_concurrency, self.settings.asr_min_interval_seconds)),
        }

    def call(self, resource: str, fn: Callable[[], T], retryable: Callable[[Exception], bool] | None = None) -> T:
        limiter = self._resources.get(resource) or self._resources["llm"]
        attempts = max(1, self.settings.api_max_retries + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return limiter.run(fn)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= attempts - 1 or not _is_retryable(exc, retryable):
                    break
                base = max(0.1, self.settings.api_retry_base_delay_seconds)
                if _is_rate_limit(exc):
                    delay = max(float(self.settings.api_rate_limit_cooldown_seconds), base * (2**attempt))
                    limiter.cooldown(delay)
                else:
                    delay = min(60.0, base * (2**attempt))
                delay += random.uniform(0.0, 0.75)
                time.sleep(delay)
        raise last_error or RuntimeError(f"{resource} call failed")


def _is_retryable(exc: Exception, custom: Callable[[Exception], bool] | None = None) -> bool:
    if custom is not None:
        return custom(exc)
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "429",
            "rate",
            "timeout",
            "temporarily",
            "503",
            "502",
            "500",
            "server disconnected",
            "connection reset",
            "connection aborted",
            "remote protocol",
            "read error",
            "connecterror",
        )
    )


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "速率限制" in text or "too many requests" in text


def is_rate_limit_error(exc: Exception) -> bool:
    return _is_rate_limit(exc)


_SCHEDULERS: dict[int, ApiScheduler] = {}
_LOCK = threading.Lock()


def get_scheduler(settings: Settings | None = None) -> ApiScheduler:
    settings = settings or get_settings()
    key = id(settings)
    with _LOCK:
        if key not in _SCHEDULERS:
            _SCHEDULERS[key] = ApiScheduler(settings)
        return _SCHEDULERS[key]
