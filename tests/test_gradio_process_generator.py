from __future__ import annotations

import time

from app.ui.gradio_app import _process_video
from app.ui.state import AppState


class FakeFuture:
    def done(self) -> bool:
        return True

    def result(self):
        class Result:
            success = False
            error = "fake failure"

        return Result()


class FakeExecutor:
    def __init__(self, max_workers: int = 1) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
        return False

    def submit(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return FakeFuture()


class FakeService:
    def process_video(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("FakeExecutor should not execute the submitted function")


def test_process_video_generator_yields_eleven_outputs(monkeypatch) -> None:
    monkeypatch.setattr("app.ui.gradio_app.ThreadPoolExecutor", FakeExecutor)
    generator = _process_video(FakeService(), None, None, AppState())

    first = next(generator)

    assert len(first) == 11


class SlowService:
    def process_video(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        time.sleep(1.2)

        class Result:
            success = False
            error = "fake failure"

        return Result()


def test_process_video_generator_emits_heartbeat_while_waiting() -> None:
    generator = _process_video(SlowService(), None, None, AppState())

    first = next(generator)
    second = next(generator)

    assert len(first) == 11
    assert "任务仍在运行" in second[2]
