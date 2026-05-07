from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.models import ConversationTurn, EvidenceGapDecision, MultimodalSegment, RefineRange, RetrievedEvidence, SummaryReport, VideoMetadata
from app.reasoning.evidence_gap_agent import EvidenceGapAgent
from app.reasoning.evidence_refinement_tool import EvidenceRefinementToolCall, build_refinement_tool_call
from app.reasoning.llm_client import LLMClient


Decision = Literal["answer_with_existing_evidence", "acquire_more_evidence", "no_refinement_needed", "early_exit"]
Relevance = Literal["strong", "weak", "none", "unknown"]


class EvidenceAcquisitionPlan(BaseModel):
    decision: Decision
    question_video_relevance: Relevance = "unknown"
    evidence_sufficiency: str = "unknown"
    reason: str = ""
    early_exit_reply: str | None = None
    actions: list[EvidenceRefinementToolCall] = Field(default_factory=list)

    @property
    def should_refine(self) -> bool:
        return self.decision == "acquire_more_evidence" and bool(self.actions)


class EvidenceGapPlanner:
    def __init__(self, settings: Settings | None = None, log_dir: str | Path | None = None) -> None:
        self.settings = settings or get_settings()
        self.log_dir = Path(log_dir) if log_dir else None

    def decide(
        self,
        question: str,
        metadata: VideoMetadata,
        segments: list[MultimodalSegment],
        turn: ConversationTurn,
        evidence: list[RetrievedEvidence],
    ) -> EvidenceAcquisitionPlan:
        mode = self.settings.evidence_gap_agent_mode.lower()
        if mode == "off":
            return EvidenceAcquisitionPlan(decision="answer_with_existing_evidence", reason="evidence_gap_agent_disabled")
        rule_decision = EvidenceGapAgent().decide(
            question,
            metadata,
            segments,
            evidence,
            turn.agreement_score,
            turn.confidence.value,
            turn.needs_visual_check,
        )
        if mode == "rule" or not self._should_use_llm():
            return _plan_from_rule(question, rule_decision)
        try:
            return self._decide_with_llm(question, metadata, segments, turn, evidence, rule_decision)
        except Exception as exc:  # noqa: BLE001
            plan = _plan_from_rule(question, rule_decision)
            plan.reason = f"llm_planner_failed_fallback_rule: {exc}; {plan.reason}"
            return plan

    def _should_use_llm(self) -> bool:
        return self.settings.has_evidence_llm_config

    def _decide_with_llm(
        self,
        question: str,
        metadata: VideoMetadata,
        segments: list[MultimodalSegment],
        turn: ConversationTurn,
        evidence: list[RetrievedEvidence],
        rule_decision: EvidenceGapDecision,
    ) -> EvidenceAcquisitionPlan:
        client = LLMClient(
            self.settings,
            self.log_dir,
            api_key=self.settings.evidence_llm_api_key,
            base_url=self.settings.runtime_evidence_llm_base_url,
            model=self.settings.runtime_evidence_llm_model,
            validate_runtime_endpoint=False,
        )
        payload = client.generate_json(
            _planner_prompt(question, metadata, _load_video_outline(self.log_dir), segments, turn, evidence, rule_decision),
            schema_hint=_schema_hint(),
        )
        return _coerce_plan(question, payload)


def _plan_from_rule(question: str, decision: EvidenceGapDecision) -> EvidenceAcquisitionPlan:
    if not decision.is_video_relevant:
        return EvidenceAcquisitionPlan(
            decision="early_exit",
            question_video_relevance="none",
            evidence_sufficiency="not_applicable",
            reason=decision.relevance_reason,
            early_exit_reply=decision.early_exit_reply,
        )
    if decision.should_refine:
        return EvidenceAcquisitionPlan(
            decision="acquire_more_evidence",
            question_video_relevance="strong",
            evidence_sufficiency="insufficient",
            reason=";".join(decision.reasons),
            actions=[build_refinement_tool_call(question, decision.target_ranges)],
        )
    return EvidenceAcquisitionPlan(
        decision="answer_with_existing_evidence",
        question_video_relevance="strong",
        evidence_sufficiency="sufficient_or_no_action",
        reason="rule_agent_no_refinement",
    )


def _coerce_plan(question: str, payload: dict[str, Any]) -> EvidenceAcquisitionPlan:
    actions: list[EvidenceRefinementToolCall] = []
    for row in payload.get("actions") or []:
        if not isinstance(row, dict) or row.get("tool") != "partial_restart_for_evidence":
            continue
        arguments = row.get("arguments") if isinstance(row.get("arguments"), dict) else {}
        arguments.setdefault("query", question)
        actions.append(EvidenceRefinementToolCall(tool="partial_restart_for_evidence", arguments=arguments))
    decision = str(payload.get("decision") or "answer_with_existing_evidence")
    if decision == "acquire_more_evidence" and not actions:
        actions = [build_refinement_tool_call(question)]
    if decision not in {"answer_with_existing_evidence", "acquire_more_evidence", "no_refinement_needed", "early_exit"}:
        decision = "answer_with_existing_evidence"
    relevance = str(payload.get("question_video_relevance") or "unknown")
    if relevance not in {"strong", "weak", "none", "unknown"}:
        relevance = "unknown"
    return EvidenceAcquisitionPlan(
        decision=decision,  # type: ignore[arg-type]
        question_video_relevance=relevance,  # type: ignore[arg-type]
        evidence_sufficiency=str(payload.get("evidence_sufficiency") or "unknown"),
        reason=str(payload.get("reason") or ""),
        early_exit_reply=payload.get("early_exit_reply") if isinstance(payload.get("early_exit_reply"), str) else None,
        actions=actions,
    )


def _load_video_outline(log_dir: Path | None) -> dict[str, Any]:
    if not log_dir:
        return {}
    summary_path = log_dir.parent / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        summary = SummaryReport.model_validate(json.loads(summary_path.read_text(encoding="utf-8")))
    except Exception:
        return {}
    return {
        "quick_overview": summary.quick_overview,
        "structured_outline": [
            {"timestamp": item.timestamp, "topic": item.topic, "key_points": item.key_points, "segment_ids": item.segment_ids}
            for item in summary.structured_outline[:12]
        ],
        "open_questions": summary.open_questions[:8],
    }


def _planner_prompt(
    question: str,
    metadata: VideoMetadata,
    outline: dict[str, Any],
    segments: list[MultimodalSegment],
    turn: ConversationTurn,
    evidence: list[RetrievedEvidence],
    rule_decision: EvidenceGapDecision,
) -> str:
    segment_brief = [
        {
            "segment_id": item.segment_id,
            "start": item.start,
            "end": item.end,
            "text": item.transcript_text[:500],
            "has_frame_caption": any(caption.caption.strip() for caption in item.visual_captions),
        }
        for item in segments[:40]
    ]
    evidence_brief = [
        {
            "evidence_id": item.evidence_id,
            "segment_id": item.segment_id,
            "evidence_type": item.evidence_type,
            "start": item.start,
            "end": item.end,
            "score": item.score,
            "text": item.text[:500],
        }
        for item in evidence[:10]
    ]
    context = {
        "task": "Decide whether the current question is relevant to this video and whether acquiring more evidence is useful.",
        "rules": [
            "Do not answer the user question.",
            "If the question is weakly related or unrelated to the video outline, choose no_refinement_needed or early_exit.",
            "If more evidence is useful, output an action using partial_restart_for_evidence.",
            "If target_ranges are unknown, leave target_ranges empty and provide a query; the tool will locate matching transcript segments by semantic retrieval.",
            "OCR is disabled; do not request OCR.",
        ],
        "available_tool": {
            "tool": "partial_restart_for_evidence",
            "arguments": {
                "query": "string, optional semantic query for locating transcript segments",
                "target_ranges": [{"start": 0.0, "end": 60.0, "reason": "string", "priority": 1}],
                "top_k_text_segments": 3,
                "context_padding_seconds": 8.0,
            },
        },
        "video_metadata": metadata.model_dump(mode="json"),
        "video_outline": outline,
        "segments_brief": segment_brief,
        "original_question": question,
        "current_turn": turn.model_dump(mode="json"),
        "current_video_evidence": evidence_brief,
        "rule_gap_decision": rule_decision.model_dump(mode="json"),
    }
    return json.dumps(context, ensure_ascii=False, indent=2)


def _schema_hint() -> str:
    return """
{
  "decision": "answer_with_existing_evidence | acquire_more_evidence | no_refinement_needed | early_exit",
  "question_video_relevance": "strong | weak | none | unknown",
  "evidence_sufficiency": "sufficient | insufficient | not_applicable | unknown",
  "reason": "short reason",
  "early_exit_reply": "optional user-facing reply",
  "actions": [
    {
      "tool": "partial_restart_for_evidence",
      "arguments": {
        "query": "semantic query used to find relevant transcript text",
        "target_ranges": [{"start": 0.0, "end": 60.0, "reason": "why this range", "priority": 1}],
        "top_k_text_segments": 3,
        "context_padding_seconds": 8.0
      }
    }
  ]
}
"""
