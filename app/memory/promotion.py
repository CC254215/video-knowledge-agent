from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings
from app.memory.mempalace_adapter import create_memory_adapter
from app.models import ConfidenceLevel, ConversationTurn, VideoMetadata

INSUFFICIENT_EVIDENCE = "当前视频证据不足以支持这个结论。"


@dataclass
class MemoryPromotionResult:
    status: str
    memory_type: str = "derived_insight"
    reason: str = ""
    provider: str = "noop"


def promote_conversation_turn(
    turn: ConversationTurn,
    metadata: VideoMetadata,
    settings: Settings | None = None,
    trace_id: str | None = None,
) -> MemoryPromotionResult:
    settings = settings or get_settings()
    if not settings.memory_promotion_enabled:
        return MemoryPromotionResult(status="skipped", reason="memory_promotion_disabled", provider=settings.mempalace_provider)
    if settings.mempalace_provider.lower() == "noop":
        return MemoryPromotionResult(status="skipped", reason="mempalace_provider_noop", provider=settings.mempalace_provider)
    if not _eligible_for_promotion(turn):
        return MemoryPromotionResult(status="skipped", reason="turn_not_eligible", provider=settings.mempalace_provider)

    adapter = None
    try:
        adapter = create_memory_adapter(settings)
        payload = _payload_for_turn(turn, metadata, trace_id)
        save = getattr(adapter, "save_derived_insight", None)
        if callable(save):
            save(payload)
        else:
            adapter.save_model_summary(payload)
        return MemoryPromotionResult(status="saved", provider=settings.mempalace_provider)
    except Exception as exc:  # noqa: BLE001
        return MemoryPromotionResult(status="failed", reason=str(exc), provider=settings.mempalace_provider)
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()


def _eligible_for_promotion(turn: ConversationTurn) -> bool:
    if INSUFFICIENT_EVIDENCE in turn.answer:
        return False
    if not turn.evidence_ids or not turn.evidence:
        return False
    if turn.confidence == ConfidenceLevel.low:
        return False
    return True


def _payload_for_turn(turn: ConversationTurn, metadata: VideoMetadata, trace_id: str | None) -> dict[str, Any]:
    return {
        "source": "video_knowledge_agent",
        "memory_type": "derived_insight",
        "summary_type": "conversation_derived_insight",
        "video_id": turn.video_id,
        "video_title": metadata.title,
        "author": metadata.author,
        "trace_id": trace_id,
        "question": turn.user_question,
        "answer": turn.answer,
        "confidence": turn.confidence.value,
        "agreement_score": turn.agreement_score,
        "evidence_ids": turn.evidence_ids,
        "query_evidence_ids": turn.query_evidence_ids,
        "answer_evidence_ids": turn.answer_evidence_ids,
        "evidence_types": turn.evidence_types,
        "timestamps": turn.timestamps,
        "evidence": [
            {
                "segment_id": item.get("segment_id"),
                "evidence_type": item.get("evidence_type"),
                "time_start": item.get("time_start"),
                "time_end": item.get("time_end"),
                "timestamp": item.get("timestamp"),
                "text": str(item.get("text") or item.get("frame_caption") or "")[:1000],
                "image_path": item.get("image_path"),
            }
            for item in turn.evidence[:8]
        ],
        "usage_rule": "Long-term memory only. It may guide future context, but current-video factual claims still require fresh video evidence.",
    }
