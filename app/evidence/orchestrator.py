from __future__ import annotations

import uuid
import shutil
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
        reanswer: Callable[..., ConversationTurn],
        persist_refinement: bool = True,
    ) -> EvidenceOrchestrationResult:
        if not _needs_evidence_planning(turn):
            return EvidenceOrchestrationResult(turn=turn, segments=segments, handled=False)

        metadata = metadata_loader()
        has_real_speech = any(
            item.transcript_text.strip() and "silent video" not in item.transcript_text.lower()
            for item in segments
        )
        trigger_reasons = _refinement_trigger_reasons(turn)
        if self.trace:
            self.trace.record("refinement_trigger_reason", {"reasons": trigger_reasons})
        if turn.evidence_plan and "speech" in turn.evidence_plan.required_modalities and not has_real_speech:
            turn.stop_reason = "required_modality_unavailable"
            turn.evidence_sufficient = False
            return EvidenceOrchestrationResult(turn=turn, segments=segments, handled=True)
        plan = EvidenceGapPlanner(self.settings, self.video_dir / "llm_logs").decide(
            question,
            metadata,  # type: ignore[arg-type]
            segments,
            turn,
            evidence,
        )
        if self.trace:
            self.trace.record(
                "evidence_acquisition_plan",
                {
                    "decision": plan.decision,
                    "question_video_relevance": plan.question_video_relevance,
                    "evidence_sufficiency": plan.evidence_sufficiency,
                    "reason": plan.reason,
                    "actions": [action.model_dump(mode="json") for action in plan.actions],
                },
            )

        if plan.decision == "early_exit":
            stopped = _early_exit_turn(
                    self.video_id,
                    question,
                    plan.early_exit_reply or "我认为这个问题与当前视频内容不强相关，因此不做额外证据补充。",
                    plan.reason,
                )
            stopped.stop_reason = "planner_early_exit"
            return EvidenceOrchestrationResult(turn=stopped, segments=segments, handled=True)

        if plan.decision == "no_refinement_needed":
            if _is_insufficient_turn(turn):
                turn = _no_refinement_turn(self.video_id, question, plan.reason)
            else:
                turn.reason = f"evidence_agent=no_refinement_needed; reason={plan.reason}; {turn.reason}"
            turn.stop_reason = "planner_no_refinement_needed"
            return EvidenceOrchestrationResult(turn=turn, segments=segments, handled=True)

        if not plan.should_refine:
            turn.stop_reason = "planner_answer_with_existing_evidence"
            return EvidenceOrchestrationResult(turn=turn, segments=segments, handled=True)

        action = plan.actions[0]
        if turn.question_analysis:
            action.arguments.temporal_scope = turn.question_analysis.temporal_scope
            action.arguments.temporal_relation = turn.question_analysis.temporal_relation
        if turn.evidence_plan:
            action.arguments.requires_global_coverage = turn.evidence_plan.requires_global_coverage
        action.arguments.has_real_speech = has_real_speech

        try:
            tool_result = execute_refinement_tool(
                self.video_id, question, self.video_dir, segments, action, self.settings,
                persist_refinement=persist_refinement,
            )
        except TypeError as exc:
            if "persist_refinement" not in str(exc):
                raise
            tool_result = execute_refinement_tool(self.video_id, question, self.video_dir, segments, action, self.settings)
        if self.trace:
            self.trace.record("evidence_tool_result", {
                "status": tool_result.status,
                "target_range_source": tool_result.target_range_source,
                "refinement_strategy": tool_result.refinement_strategy,
                "refined_frame_count": tool_result.added_frames,
                "resolved_ranges": [item.model_dump(mode="json") for item in tool_result.resolved_ranges],
                "added_captions": tool_result.added_captions,
                "persist_refinement": persist_refinement,
                "temporary_workspace": bool(tool_result.temporary_workspace),
                "error": tool_result.error,
            })
        if tool_result.status == "failed":
            turn.reason = f"evidence_tool_failed={tool_result.error}; {turn.reason}"
            return EvidenceOrchestrationResult(turn=turn, segments=segments, handled=False)
        if persist_refinement:
            updated_segments = self._load_segments()
            self.store.add_multimodal_segments(self.video_id, updated_segments)
            temporary_store = None
        else:
            updated_segments = tool_result.temporary_segments or segments
            temporary_store = tool_result.temporary_store
        try:
            refined_turn = reanswer(updated_segments, True, temporary_store)
        except TypeError:
            # Keep lightweight test doubles and legacy adapters source-compatible.
            refined_turn = reanswer(updated_segments, True)
        if tool_result.status == "no_target_ranges":
            refined_turn.stop_reason = "no_target_ranges"
        elif tool_result.status == "no_new_evidence":
            refined_turn.stop_reason = "no_new_evidence"
        elif refined_turn.evidence_sufficient:
            refined_turn.stop_reason = "refinement_succeeded"
        else:
            refined_turn.stop_reason = "insufficient_after_single_refinement"
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
        if not persist_refinement and tool_result.temporary_workspace:
            shutil.rmtree(tool_result.temporary_workspace, ignore_errors=True)
        return EvidenceOrchestrationResult(turn=refined_turn, segments=updated_segments, handled=True)

    def _load_segments(self) -> list[MultimodalSegment]:
        path = self.video_dir / "multimodal_segments.json"
        return load_multimodal_segments(path) if path.exists() else []


def _is_insufficient_turn(turn: ConversationTurn) -> bool:
    return INSUFFICIENT_EVIDENCE in turn.answer or not turn.evidence


def _needs_evidence_planning(turn: ConversationTurn) -> bool:
    if turn.evidence_gate_result and turn.evidence_gate_result.status == "supported_with_scope_limit":
        # Scope uncertainty is not a claim failure. Best-effort answers should
        # be returned with a limitation instead of triggering another scan.
        return any(row.get("status") in {"unsupported", "contradicted"} for row in turn.fact_support)
    return bool(
        turn.structured_evidence_gate.get("unmet_constraints")
        or any(row.get("status") != "supported" for row in turn.fact_support)
        or not turn.evidence
        or turn.confidence == ConfidenceLevel.low
        or _is_insufficient_turn(turn)
    )


def _refinement_trigger_reasons(turn: ConversationTurn) -> list[str]:
    reasons = list(turn.structured_evidence_gate.get("unmet_constraints", []))
    reasons.extend(
        f"{row.get('kind')} unsupported: {row.get('fact')}"
        for row in turn.fact_support
        if row.get("status") != "supported"
    )
    if not turn.evidence:
        reasons.append("no_evidence")
    if turn.confidence == ConfidenceLevel.low:
        reasons.append("low_confidence")
    if _is_insufficient_turn(turn):
        reasons.append("legacy_insufficient_answer")
    return list(dict.fromkeys(reasons))


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
        evidence_sufficient=False,
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
        evidence_sufficient=False,
        reason=f"evidence_agent=no_refinement_needed; reason={reason}",
        suggested_followup_questions=["可以问我视频的核心观点、某个时间点内容，或让我列出证据片段。"],
    )
