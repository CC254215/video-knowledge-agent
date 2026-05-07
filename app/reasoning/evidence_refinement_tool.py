from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.config import Settings, get_settings
from app.models import MultimodalSegment, RefineRange, RetrievedEvidence, VideoMetadata
from app.retrieval.chroma_memory import ChromaMemoryStore
from app.retrieval.hybrid_retriever import HybridEvidenceRetriever


REFINE_TOOL_NAME = "partial_restart_for_evidence"


class EvidenceRefinementToolArguments(BaseModel):
    query: str | None = None
    target_ranges: list[RefineRange] = Field(default_factory=list)
    top_k_text_segments: int = 3
    max_ranges: int = 3
    context_padding_seconds: float = 8.0

    @field_validator("top_k_text_segments", "max_ranges")
    @classmethod
    def positive_int(cls, value: int) -> int:
        return max(1, int(value))

    @field_validator("context_padding_seconds")
    @classmethod
    def non_negative_padding(cls, value: float) -> float:
        return max(0.0, float(value))


class EvidenceRefinementToolCall(BaseModel):
    tool: Literal["partial_restart_for_evidence"]
    arguments: EvidenceRefinementToolArguments = Field(default_factory=EvidenceRefinementToolArguments)


@dataclass
class EvidenceRefinementToolResult:
    status: str
    tool: str = REFINE_TOOL_NAME
    resolved_ranges: list[RefineRange] = field(default_factory=list)
    source_evidence_ids: list[str] = field(default_factory=list)
    added_frames: int = 0
    added_text_segments: int = 0
    added_captions: int = 0
    error: str | None = None


def build_refinement_tool_call(
    question: str,
    target_ranges: list[RefineRange] | None = None,
    query: str | None = None,
) -> EvidenceRefinementToolCall:
    return EvidenceRefinementToolCall(
        tool=REFINE_TOOL_NAME,
        arguments=EvidenceRefinementToolArguments(query=query or question, target_ranges=target_ranges or []),
    )


def execute_refinement_tool(
    video_id: str,
    question: str,
    video_dir: str | Path,
    segments: list[MultimodalSegment],
    tool_call: EvidenceRefinementToolCall,
    settings: Settings | None = None,
) -> EvidenceRefinementToolResult:
    settings = settings or get_settings()
    metadata = _load_metadata(Path(video_dir), video_id)
    resolved_ranges, source_evidence = resolve_refinement_ranges(
        video_id=video_id,
        question=question,
        segments=segments,
        settings=settings,
        metadata=metadata,
        arguments=tool_call.arguments,
    )
    if not resolved_ranges:
        return EvidenceRefinementToolResult(status="no_target_ranges", resolved_ranges=[], source_evidence_ids=[])

    try:
        from app import pipeline

        result = pipeline.partial_restart_for_evidence(video_id, resolved_ranges, question, settings)
        return EvidenceRefinementToolResult(
            status=str(result.status),
            resolved_ranges=resolved_ranges,
            source_evidence_ids=[item.evidence_id for item in source_evidence],
            added_frames=result.added_frames,
            added_text_segments=result.added_text_segments,
            added_captions=result.added_captions,
        )
    except Exception as exc:  # noqa: BLE001
        return EvidenceRefinementToolResult(
            status="failed",
            resolved_ranges=resolved_ranges,
            source_evidence_ids=[item.evidence_id for item in source_evidence],
            error=str(exc),
        )


def resolve_refinement_ranges(
    video_id: str,
    question: str,
    segments: list[MultimodalSegment],
    settings: Settings,
    metadata: VideoMetadata,
    arguments: EvidenceRefinementToolArguments,
) -> tuple[list[RefineRange], list[RetrievedEvidence]]:
    explicit_ranges = _sanitize_ranges(arguments.target_ranges, metadata.duration, arguments.max_ranges)
    if explicit_ranges:
        return explicit_ranges, []

    query = arguments.query or question
    source_evidence = _retrieve_text_evidence(video_id, query, segments, settings, arguments.top_k_text_segments)
    ranges = [
        RefineRange(
            start=max(0.0, item.start - arguments.context_padding_seconds),
            end=_clamp_end(item.end + arguments.context_padding_seconds, metadata.duration),
            reason=f"semantic_text_match:{item.evidence_id}",
            priority=index + 1,
        )
        for index, item in enumerate(source_evidence)
        if item.end >= item.start
    ]
    return _sanitize_ranges(_merge_ranges(ranges), metadata.duration, arguments.max_ranges), source_evidence


def tool_result_to_reason(result: EvidenceRefinementToolResult) -> str:
    ranges = ", ".join(f"{_format_seconds(item.start)}-{_format_seconds(item.end)}" for item in result.resolved_ranges)
    return (
        f"refinement_tool={result.tool}; status={result.status}; ranges=[{ranges}]; "
        f"source_evidence_ids={result.source_evidence_ids}; added_frames={result.added_frames}; "
        f"added_text_segments={result.added_text_segments}; added_captions={result.added_captions}"
    )


def _retrieve_text_evidence(
    video_id: str,
    query: str,
    segments: list[MultimodalSegment],
    settings: Settings,
    top_k: int,
) -> list[RetrievedEvidence]:
    store = ChromaMemoryStore(settings)
    store.add_multimodal_segments(video_id, segments)
    retriever = HybridEvidenceRetriever(store, settings)
    return retriever.search(video_id, query, segments, top_k=top_k, evidence_types=["speech"])


def _sanitize_ranges(ranges: list[RefineRange], duration: float | None, max_ranges: int) -> list[RefineRange]:
    output: list[RefineRange] = []
    for item in sorted(ranges, key=lambda row: (row.priority, row.start, row.end)):
        start = max(0.0, float(item.start))
        end = _clamp_end(max(start, float(item.end)), duration)
        if end <= start:
            end = _clamp_end(start + 1.0, duration)
        output.append(RefineRange(start=start, end=end, reason=item.reason, priority=item.priority))
        if len(output) >= max_ranges:
            break
    return output


def _merge_ranges(ranges: list[RefineRange]) -> list[RefineRange]:
    merged: list[RefineRange] = []
    for item in sorted(ranges, key=lambda row: (row.start, row.end)):
        if not merged or item.start > merged[-1].end + 2:
            merged.append(item)
            continue
        previous = merged[-1]
        merged[-1] = RefineRange(
            start=previous.start,
            end=max(previous.end, item.end),
            reason=f"{previous.reason};{item.reason}",
            priority=min(previous.priority, item.priority),
        )
    return merged


def _clamp_end(end: float, duration: float | None) -> float:
    if duration is None or duration <= 0:
        return end
    return min(float(duration), end)


def _load_metadata(video_dir: Path, video_id: str) -> VideoMetadata:
    path = video_dir / "metadata.json"
    if path.exists():
        return VideoMetadata.model_validate(json.loads(path.read_text(encoding="utf-8")))
    return VideoMetadata(video_id=video_id, title=video_id)


def _format_seconds(value: float) -> str:
    total = max(0, int(round(value)))
    return f"{total // 60:02d}:{total % 60:02d}"
