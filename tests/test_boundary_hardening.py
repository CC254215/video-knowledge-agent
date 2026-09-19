from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from app.api.task_store import APITaskStore, MAX_PROGRESS_EVENTS
from app.config import Settings
from app.evals.benchmark import EvalOptions, evaluate_dataset
from app.ingestion.url_safety import UnsafeMediaURLError, validate_public_media_url
from app.models import ConfidenceLevel, ConversationTurn, MultimodalSegment
from app.reasoning.conversation_agent import ConversationAgent
from app.retrieval.chroma_memory import ChromaMemoryStore
from app.runtime.processing_state import ProcessingState
from app.transcript.multimodal_segmenter import save_multimodal_segments


def _turn(turn_id: str, conversation_id: str | None = None) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        video_id="v1",
        conversation_id=conversation_id,
        user_question="question",
        answer="当前视频证据不足以支持这个结论。",
        confidence=ConfidenceLevel.low,
        needs_visual_check=False,
        reason="test",
    )


def test_url_guard_rejects_private_and_non_http(monkeypatch) -> None:
    with pytest.raises(UnsafeMediaURLError):
        validate_public_media_url("file:///tmp/video.mp4")
    with pytest.raises(UnsafeMediaURLError):
        validate_public_media_url("http://127.0.0.1/video.mp4")

    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(socket.AF_INET, 0, 0, "", ("10.0.0.8", 80))])
    with pytest.raises(UnsafeMediaURLError):
        validate_public_media_url("https://media.example/video.mp4")


def test_url_guard_accepts_official_youtube_hosts_without_dns(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: (_ for _ in ()).throw(socket.gaierror()))
    assert validate_public_media_url("https://www.youtube.com/watch?v=abc123")
    assert validate_public_media_url("https://youtu.be/abc123")
    with pytest.raises(UnsafeMediaURLError):
        validate_public_media_url("https://youtube.com.attacker.example/video")


def test_conversation_histories_are_isolated_and_atomic(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path, chroma_path=tmp_path / "chroma")
    agent = ConversationAgent("v1", tmp_path / "video", settings=settings, store=ChromaMemoryStore(settings, use_persistent=False))

    agent._append_turn(_turn("t1", "conversation-a"), "conversation-a")
    agent._append_turn(_turn("t2", "conversation-b"), "conversation-b")

    path_a = agent._history_path("conversation-a")
    path_b = agent._history_path("conversation-b")
    assert path_a != path_b
    assert json.loads(path_a.read_text(encoding="utf-8"))[0]["turn_id"] == "t1"
    assert json.loads(path_b.read_text(encoding="utf-8"))[0]["turn_id"] == "t2"
    assert not list(path_a.parent.glob("*.tmp"))


def test_dry_run_answer_does_not_write_history_or_trace(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path, chroma_path=tmp_path / "chroma", memory_promotion_enabled=False)
    video_dir = tmp_path / "videos" / "v1"
    video_dir.mkdir(parents=True)
    save_multimodal_segments(
        [MultimodalSegment(segment_id="seg_0000", start=0, end=10, transcript_text="可追溯证据")],
        video_dir,
    )
    store = ChromaMemoryStore(settings, use_persistent=False)

    turn = ConversationAgent("v1", video_dir, settings=settings, store=store).answer(
        "核心观点是什么？",
        conversation_id="eval-case",
        persist=False,
    )

    assert turn.conversation_id == "eval-case"
    assert not (video_dir / "qa_history").exists()
    assert not (video_dir / "qa_traces").exists()


def test_task_progress_is_bounded(tmp_path: Path) -> None:
    store = APITaskStore(tmp_path / "tasks.sqlite3")
    store.create_task("t1", "process", {})
    store.mark_running("t1")
    for index in range(MAX_PROGRESS_EVENTS + 25):
        store.add_progress("t1", str(index))
    progress = store.get_task("t1")["progress"]
    assert len(progress) == MAX_PROGRESS_EVENTS
    assert progress[0]["message"] == "25"


def test_memory_index_removes_stale_documents(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path, chroma_path=tmp_path / "chroma")
    store = ChromaMemoryStore(settings, use_persistent=False)
    first = MultimodalSegment(segment_id="seg_old", start=0, end=1, transcript_text="obsolete evidence")
    second = MultimodalSegment(segment_id="seg_new", start=1, end=2, transcript_text="current evidence")
    store.add_multimodal_segments("v1", [first, second])
    store.add_multimodal_segments("v1", [second])
    assert all(item.segment_id != "seg_old" for item in store.search("v1", "obsolete", top_k=10))


def test_processing_state_requires_matching_fingerprint(tmp_path: Path) -> None:
    artifact = tmp_path / "summary.json"
    artifact.write_text("{}", encoding="utf-8")
    state = ProcessingState(tmp_path)
    state.start("summary")
    state.succeed("summary", [artifact], {"fingerprint": "v1"})
    assert state.is_succeeded("summary", [artifact], fingerprint="v1")
    assert not state.is_succeeded("summary", [artifact], fingerprint="v2")


def test_empty_eval_dataset_fails_and_retrieval_only_omits_answer_metrics(tmp_path: Path, monkeypatch) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    empty_report = evaluate_dataset(empty, Settings(_env_file=None, data_dir=tmp_path), EvalOptions())
    assert empty_report["ok"] is False
    assert empty_report["error"] == "empty_dataset"

    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps([{"video_id": "v1", "question": "q", "gold_segment_ids": ["seg_0"]}]), encoding="utf-8")
    segment = MultimodalSegment(segment_id="seg_0", start=0, end=1, transcript_text="q answer")
    monkeypatch.setattr("app.pipeline.read_multimodal_segments", lambda video_id, settings: [segment])
    report = evaluate_dataset(dataset, Settings(_env_file=None, data_dir=tmp_path), EvalOptions(run_answers=False))
    assert "grounding_precision" not in report["aggregate"]
    assert "answer_contains_recall" not in report["aggregate"]


def test_env_file_is_not_reloaded_when_disabled(monkeypatch) -> None:
    for key in ("LLM_API_KEY", "ZHIPU_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert Settings(_env_file=None).openai_api_key is None
