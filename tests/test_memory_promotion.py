from __future__ import annotations

from app.config import Settings
from app.memory import promotion
from app.memory.promotion import promote_conversation_turn
from app.models import ConfidenceLevel, ConversationTurn, VideoMetadata


class RecordingAdapter:
    def __init__(self) -> None:
        self.derived = []
        self.conversations = []

    def save_conversation_turn(self, payload):
        self.conversations.append(payload)

    def save_derived_insight(self, payload):
        self.derived.append(payload)

    def save_model_summary(self, payload):
        raise AssertionError("derived insight should be preferred")

    def save_raw_evidence(self, payload):
        pass

    def save_user_verified_insight(self, payload):
        pass

    def search_related_memories(self, query: str, limit: int = 5):
        return []

    def close(self):
        pass


def test_promote_conversation_turn_saves_grounded_answer(monkeypatch) -> None:
    adapter = RecordingAdapter()
    monkeypatch.setattr(promotion, "create_memory_adapter", lambda settings: adapter)

    result = promote_conversation_turn(
        _turn(confidence=ConfidenceLevel.medium),
        VideoMetadata(video_id="v1", title="Video A", author="author"),
        Settings(_env_file=None, mempalace_provider="mcp_stdio", mempalace_auto_status=False),
        trace_id="trace-1",
    )

    assert result.status == "saved"
    assert result.memory_type == "conversation_turn+derived_insight"
    assert adapter.conversations[0]["content"] == "User:\nWhat is the core point?\n\nAssistant:\nThe core point is grounded in evidence."
    assert adapter.derived
    payload = adapter.derived[0]
    assert payload["memory_type"] == "derived_insight"
    assert payload["summary_type"] == "conversation_derived_insight"
    assert payload["trace_id"] == "trace-1"
    assert payload["evidence_ids"] == ["seg_0"]


def test_promote_conversation_turn_skips_low_confidence(monkeypatch) -> None:
    adapter = RecordingAdapter()
    monkeypatch.setattr(promotion, "create_memory_adapter", lambda settings: adapter)

    result = promote_conversation_turn(
        _turn(confidence=ConfidenceLevel.low),
        VideoMetadata(video_id="v1", title="Video A"),
        Settings(_env_file=None, mempalace_provider="mcp_stdio", mempalace_auto_status=False),
    )

    assert result.status == "saved"
    assert result.memory_type == "conversation_turn"
    assert result.reason == "raw_conversation_saved;derived_insight_not_eligible"
    assert len(adapter.conversations) == 1
    assert adapter.derived == []


def test_promote_conversation_turn_skips_noop_provider() -> None:
    result = promote_conversation_turn(
        _turn(confidence=ConfidenceLevel.high),
        VideoMetadata(video_id="v1", title="Video A"),
        Settings(_env_file=None, mempalace_provider="noop"),
    )

    assert result.status == "skipped"
    assert result.reason == "mempalace_provider_noop"


def _turn(confidence: ConfidenceLevel) -> ConversationTurn:
    return ConversationTurn(
        turn_id="t1",
        video_id="v1",
        user_question="What is the core point?",
        answer="The core point is grounded in evidence.",
        evidence_ids=["seg_0"],
        evidence=[
            {
                "segment_id": "seg_0",
                "evidence_type": "speech",
                "time_start": "00:00:00",
                "time_end": "00:00:10",
                "timestamp": "00:00:00",
                "text": "speech evidence",
            }
        ],
        timestamps=["00:00:00"],
        evidence_types=["speech"],
        confidence=confidence,
        agreement_score=0.8,
        query_evidence_ids=["v1:seg_0:speech"],
        answer_evidence_ids=["v1:seg_0:speech"],
        needs_visual_check=False,
        reason="test",
    )
