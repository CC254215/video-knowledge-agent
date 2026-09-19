from __future__ import annotations

import json
import tempfile
import uuid
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
    has_real_speech: bool | None = None
    temporal_scope: str = "none"
    temporal_relation: str = "none"
    requires_global_coverage: bool = False

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
    target_range_source: str = "unknown"
    refinement_strategy: str = "local_first"
    error: str | None = None
    temporary_segments: list[MultimodalSegment] | None = field(default=None, repr=False)
    temporary_store: ChromaMemoryStore | None = field(default=None, repr=False)
    temporary_workspace: str | None = field(default=None, repr=False)


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
    persist_refinement: bool = True,
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
        use_persistent_store=persist_refinement,
    )
    if not resolved_ranges:
        return EvidenceRefinementToolResult(status="no_target_ranges", resolved_ranges=[], source_evidence_ids=[], target_range_source="none")

    try:
        from app import pipeline

        refinement_strategy = _refinement_strategy(tool_call.arguments)
        if not persist_refinement:
            return _execute_temporary_refinement(
                video_id, question, video_dir, segments, resolved_ranges, metadata,
                refinement_strategy, source_evidence, settings,
            )
        if refinement_strategy == "local_first":
            result = pipeline.partial_restart_for_evidence(video_id, resolved_ranges, question, settings)
        else:
            result = pipeline.partial_restart_for_evidence(
                video_id,
                resolved_ranges,
                question,
                settings,
                refinement_strategy=refinement_strategy,
            )
        return EvidenceRefinementToolResult(
            status=str(result.status),
            resolved_ranges=resolved_ranges,
            source_evidence_ids=[item.evidence_id for item in source_evidence],
            added_frames=result.added_frames,
            added_text_segments=result.added_text_segments,
            added_captions=result.added_captions,
            target_range_source=_target_range_source(tool_call.arguments, source_evidence),
            refinement_strategy=refinement_strategy,
        )
    except Exception as exc:  # noqa: BLE001
        return EvidenceRefinementToolResult(
            status="failed",
            resolved_ranges=resolved_ranges,
            source_evidence_ids=[item.evidence_id for item in source_evidence],
            target_range_source=_target_range_source(tool_call.arguments, source_evidence),
            refinement_strategy=_refinement_strategy(tool_call.arguments),
            error=str(exc),
        )


def _execute_temporary_refinement(
    video_id: str,
    question: str,
    video_dir: str | Path,
    segments: list[MultimodalSegment],
    ranges: list[RefineRange],
    metadata: VideoMetadata,
    refinement_strategy: str,
    source_evidence: list[RetrievedEvidence],
    settings: Settings,
) -> EvidenceRefinementToolResult:
    """Acquire evidence in an isolated workspace for the current QA turn."""
    from app import pipeline
    workspace = Path(tempfile.mkdtemp(prefix=f"vka-refine-{video_id}-"))
    # Frame extraction needs metadata, while the source video remains in the
    # original directory.  Nothing under the original video directory is
    # written in this branch.
    (workspace / "metadata.json").write_text(metadata.model_dump_json(), encoding="utf-8")
    working = [item.model_copy(deep=True) for item in segments]
    transcript_segments = pipeline.read_segments(video_id, settings)
    before_frames = {fid for seg in working for fid in seg.representative_frame_ids}
    before_captions = {cap.frame_id for seg in working for cap in seg.visual_captions if cap.caption.strip()}
    before_text = {seg.segment_id for seg in working if seg.transcript_text.strip() and any(_overlaps(seg.start, seg.end, r.start, r.end) for r in ranges)}
    refinement_id = uuid.uuid4().hex
    refiner = __import__("app.vision.frame_refiner", fromlist=["FrameEvidenceRefiner"]).FrameEvidenceRefiner(workspace, settings=settings)
    for range_index, item in enumerate(ranges[:3]):
        working = pipeline._ensure_refined_text_segment(video_id, working, transcript_segments, item)
        working = refiner.refine(
            video_id, question, (item.start, item.end), working,
            refinement_strategy=refinement_strategy, refinement_id=refinement_id,
            range_index=range_index, persist=False, metadata=metadata,
        )
    after_frames = {fid for seg in working for fid in seg.representative_frame_ids}
    after_captions = {cap.frame_id for seg in working for cap in seg.visual_captions if cap.caption.strip()}
    after_text = {seg.segment_id for seg in working if seg.transcript_text.strip() and any(_overlaps(seg.start, seg.end, r.start, r.end) for r in ranges)}
    added_frames = sorted(after_frames - before_frames)
    added_captions = sorted(after_captions - before_captions)
    added_text = sorted(after_text - before_text)
    added_ids = [f"{video_id}:{sid}:speech" for sid in added_text]
    added_ids.extend(f"{video_id}:frame:{fid}:caption" for fid in added_captions)
    temp_store = ChromaMemoryStore(settings, use_persistent=False)
    temp_store.add_multimodal_segments(video_id, working)
    return EvidenceRefinementToolResult(
        status="updated" if added_frames or added_captions or added_text else "no_new_evidence",
        resolved_ranges=ranges,
        source_evidence_ids=[item.evidence_id for item in source_evidence],
        added_frames=len(added_frames), added_text_segments=len(added_text), added_captions=len(added_captions),
        target_range_source="planner" if not source_evidence else "speech_guided",
        refinement_strategy=refinement_strategy,
        temporary_segments=working, temporary_store=temp_store, temporary_workspace=str(workspace),
    )


def resolve_refinement_ranges(
    video_id: str,
    question: str,
    segments: list[MultimodalSegment],
    settings: Settings,
    metadata: VideoMetadata,
    arguments: EvidenceRefinementToolArguments,
    use_persistent_store: bool = True,
) -> tuple[list[RefineRange], list[RetrievedEvidence]]:
    explicit_ranges = _sanitize_ranges(arguments.target_ranges, metadata.duration, arguments.max_ranges)
    if explicit_ranges:
        # Planner-supplied visual ranges must not suppress speech evidence.
        # Retrieve transcript hits independently and use them to widen/anchor
        # the refinement set when they overlap the requested ranges.
        source_evidence = _retrieve_text_evidence(
            video_id, arguments.query or question, segments, settings,
            arguments.top_k_text_segments, use_persistent_store=use_persistent_store,
        )
        overlapping = [
            item for item in source_evidence
            if any(item.end >= target.start and item.start <= target.end for target in explicit_ranges)
        ]
        return explicit_ranges, overlapping

    has_real_speech = arguments.has_real_speech
    if has_real_speech is None:
        has_real_speech = any(
            item.transcript_text.strip() and "silent video" not in item.transcript_text.lower()
            for item in segments
        )
    if arguments.requires_global_coverage or arguments.temporal_scope == "whole_video" or not has_real_speech:
        return _coverage_ranges(metadata.duration, segments, arguments.max_ranges), []

    query = arguments.query or question
    source_evidence = _retrieve_text_evidence(video_id, query, segments, settings, arguments.top_k_text_segments, use_persistent_store=use_persistent_store)
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
        f"added_text_segments={result.added_text_segments}; added_captions={result.added_captions}; "
        f"target_range_source={result.target_range_source}; refinement_strategy={result.refinement_strategy}; "
        f"persistence={'temporary' if result.temporary_segments is not None else 'persistent'}"
    )


def _coverage_ranges(duration: float | None, segments: list[MultimodalSegment], max_ranges: int) -> list[RefineRange]:
    total = float(duration or 0.0)
    if total <= 0 and segments:
        total = max(item.end for item in segments)
    if total <= 0:
        return []
    count = min(3, max_ranges)
    width = total / count
    labels = ["early", "middle", "late"]
    return [
        RefineRange(
            start=index * width,
            end=total if index == count - 1 else (index + 1) * width,
            reason=f"coverage_fallback:{labels[index]}",
            priority=index + 1,
        )
        for index in range(count)
    ]


def _target_range_source(arguments: EvidenceRefinementToolArguments, source_evidence: list[RetrievedEvidence]) -> str:
    if arguments.target_ranges:
        return "planner"
    if source_evidence:
        return "speech_guided"
    if arguments.requires_global_coverage or arguments.temporal_scope == "whole_video" or arguments.has_real_speech is False:
        return "coverage_fallback"
    return "temporal_fallback"


def _refinement_strategy(arguments: EvidenceRefinementToolArguments) -> str:
    if arguments.requires_global_coverage or (
        arguments.temporal_scope == "whole_video" and arguments.temporal_relation in {"stage", "sequence"}
    ):
        return "temporal_stratified"
    return "local_first"


def _retrieve_text_evidence(
    video_id: str,
    query: str,
    segments: list[MultimodalSegment],
    settings: Settings,
    top_k: int,
    use_persistent_store: bool = True,
) -> list[RetrievedEvidence]:
    store = ChromaMemoryStore(settings, use_persistent=use_persistent_store)
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
