from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, get_settings
from app import pipeline
from app.models import ConversationTurn


@dataclass
class ChatAnswerResult:
    answer: str
    evidence: list[dict]
    timestamps: list[str]
    evidence_types: list[str]
    confidence: str
    needs_visual_check: bool
    reason: str
    suggested_followup_questions: list[str]
    raw: ConversationTurn | None = None
    error: str | None = None


class ChatService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def ask(self, video_id: str | None, question: str, conversation_id: str | None = None) -> ChatAnswerResult:
        if not video_id:
            return ChatAnswerResult(
                answer="请先输入 URL 或上传视频并完成处理。",
                evidence=[],
                timestamps=[],
                evidence_types=[],
                confidence="low",
                needs_visual_check=False,
                reason="当前没有已处理的视频上下文。",
                suggested_followup_questions=["上传本地视频", "输入公开视频 URL"],
                raw=None,
                error="missing_video",
            )
        if not question.strip():
            return ChatAnswerResult(
                answer="请输入问题。",
                evidence=[],
                timestamps=[],
                evidence_types=[],
                confidence="low",
                needs_visual_check=False,
                reason="empty_question",
                suggested_followup_questions=[],
            )
        try:
            turn = pipeline.chat(video_id, question, conversation_id=conversation_id, settings=self.settings)
            return ChatAnswerResult(
                answer=turn.answer,
                evidence=turn.evidence,
                timestamps=turn.timestamps,
                evidence_types=[str(item) for item in turn.evidence_types],
                confidence=turn.confidence.value,
                needs_visual_check=turn.needs_visual_check,
                reason=turn.reason,
                suggested_followup_questions=turn.suggested_followup_questions,
                raw=turn,
            )
        except Exception as exc:  # noqa: BLE001
            return ChatAnswerResult(
                answer="处理问题时失败，请检查视频是否已完整处理。",
                evidence=[],
                timestamps=[],
                evidence_types=[],
                confidence="low",
                needs_visual_check=False,
                reason=str(exc).splitlines()[0][:1000],
                suggested_followup_questions=[],
                error=str(exc),
            )
