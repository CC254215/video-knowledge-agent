from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.config import Settings
from app.models import ConfidenceLevel, ConversationTurn, MultimodalSegment, RetrievedEvidence
from app.reasoning.evidence_gap_planner import EvidenceGapPlanner
from app.reasoning.evidence_refinement_tool import execute_refinement_tool, tool_result_to_reason
from app.reasoning.summarizer import format_timestamp
from app.retrieval.chroma_memory import ChromaMemoryStore
from app.transcript.multimodal_segmenter import load_multimodal_segments
from app.evidence.trace import EvidenceTraceRecorder

INSUFFICIENT_EVIDENCE = "当前视频证据不足以支持这个结论。"


@dataclass
class EvidenceOrchestrationResult:
    turn: ConversationTurn
    segments: list[MultimodalSegment]
    handled: bool = False


class EvidenceOrchestrator:
    """Coordinate evidence sufficiency planning and evidence acquisition tools."""

    def __init__(
        self,
        video_id: str,
        video_dir: str | Path,
        settings: Settings,
        store: ChromaMemoryStore,
        trace: EvidenceTraceRecorder | None = None,
    ) -> None:
        self.video_id = video_id
        self.video_dir = Path(video_dir)
        self.settings = settings
        self.store = store
        self.trace = trace

    def review_and_maybe_refine(
        self,
        question: str,
        turn: ConversationTurn,
        segments: list[MultimodalSegment],
        evidence: list[RetrievedEvidence],
        metadata_loader: Callable[[], object],
        reanswer: Callable[[list[MultimodalSegment], bool], ConversationTurn],
    ) -> EvidenceOrchestrationResult:
        if not _needs_evidence_planning(turn):
            return EvidenceOrchestrationResult(turn=turn, segments=segments, handled=False)

        plan = EvidenceGapPlanner(self.settings, self.video_dir / "llm_logs").decide(
            question,
            metadata_loader(),  # type: ignore[arg-type]
            segments,
            turn,
            evidence,
        )
        if self.trace:
            self.trace.record(
                "evidence_plan",
                {
                    "decision": plan.decision,
                    "question_video_relevance": plan.question_video_relevance,
                    "evidence_sufficiency": plan.evidence_sufficiency,
                    "reason": plan.reason,
                    "actions": [action.model_dump(mode="json") for action in plan.actions],
                },
            )

        if plan.decision == "early_exit":
            return EvidenceOrchestrationResult(
                turn=_early_exit_turn(
                    self.video_id,
                    question,
                    plan.early_exit_reply or "我认为这个问题与当前视频内容不强相关，因此不做额外证据补充。",
                    plan.reason,
                ),
                segments=segments,
                handled=True,
            )

        if plan.decision == "no_refinement_needed":
            if _is_insufficient_turn(turn):
                turn = _no_refinement_turn(self.video_id, question, plan.reason)
            else:
                turn.reason = f"evidence_agent=no_refinement_needed; reason={plan.reason}; {turn.reason}"
            return EvidenceOrchestrationResult(turn=turn, segments=segments, handled=True)

        if not plan.should_refine:
            return EvidenceOrchestrationResult(turn=turn, segments=segments, handled=True)

        tool_result = execute_refinement_tool(
            self.video_id,
            question,
            self.video_dir,
            segments,
            plan.actions[0],
            self.settings,
        )
        if self.trace:
            self.trace.record("evidence_tool_result", {"result": tool_result})
        updated_segments = self._load_segments()
        self.store.add_multimodal_segments(self.video_id, updated_segments)
        refined_turn = reanswer(updated_segments, True)
        range_text = ", ".join(format_timestamp(item.start) + "-" + format_timestamp(item.end) for item in tool_result.resolved_ranges)
        if tool_result.added_frames or tool_result.added_text_segments or tool_result.added_captions:
            refined_turn.reason = (
                f"gap_agent=enabled; pipeline_restart=partial; ranges=[{range_text}]; "
                f"{tool_result_to_reason(tool_result)}; gap_agent_round4=triggered; {refined_turn.reason}"
            )
        else:
            refined_turn.reason = (
                f"gap_agent=enabled; refine_executed=true; added=0; fallback=graceful_degrade; "
                f"ranges=[{range_text}]; {tool_result_to_reason(tool_result)}; {refined_turn.reason}"
            )
        return EvidenceOrchestrationResult(turn=refined_turn, segments=updated_segments, handled=True)

    def _load_segments(self) -> list[MultimodalSegment]:
        path = self.video_dir / "multimodal_segments.json"
        return load_multimodal_segments(path) if path.exists() else []


def _is_insufficient_turn(turn: ConversationTurn) -> bool:
    return INSUFFICIENT_EVIDENCE in turn.answer or not turn.evidence


def _needs_evidence_planning(turn: ConversationTurn) -> bool:
    return _is_insufficient_turn(turn) or turn.confidence == ConfidenceLevel.low


def _early_exit_turn(video_id: str, question: str, reply: str, reason: str) -> ConversationTurn:
    return ConversationTurn(
        turn_id=str(uuid.uuid4()),
        video_id=video_id,
        user_question=question,
        answer=reply,
        evidence_ids=[],
        evidence=[],
        timestamps=[],
        evidence_types=[],
        confidence=ConfidenceLevel.low,
        agreement_score=0.0,
        query_evidence_ids=[],
        answer_evidence_ids=[],
        dpp_used=False,
        dpp_details={},
        needs_visual_check=False,
        reason=f"gap_agent=early_exit; reason={reason}",
        suggested_followup_questions=["你可以问我这个视频的核心观点、某个时间点内容，或让我列出证据片段。"],
    )


def _no_refinement_turn(video_id: str, question: str, reason: str) -> ConversationTurn:
    return ConversationTurn(
        turn_id=str(uuid.uuid4()),
        video_id=video_id,
        user_question=question,
        answer="我认为这个问题与当前视频内容不强相关，因此不做额外证据补充。你可以换一个更贴近视频内容的问题，或指定你想追问的视频片段。",
        evidence_ids=[],
        evidence=[],
        timestamps=[],
        evidence_types=[],
        confidence=ConfidenceLevel.low,
        agreement_score=0.0,
        query_evidence_ids=[],
        answer_evidence_ids=[],
        dpp_used=False,
        dpp_details={},
        needs_visual_check=False,
        reason=f"evidence_agent=no_refinement_needed; reason={reason}",
        suggested_followup_questions=["可以问我视频的核心观点、某个时间点内容，或让我列出证据片段。"],
    )
