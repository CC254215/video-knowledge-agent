from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.models import MultimodalSegment, RefineRange, VideoMetadata
from app.reasoning.evidence_refinement_tool import (
    EvidenceRefinementToolCall,
    EvidenceRefinementToolResult,
    build_refinement_tool_call,
    execute_refinement_tool,
    resolve_refinement_ranges,
)


def test_refinement_tool_resolves_ranges_from_semantic_text_match(tmp_path: Path) -> None:
    segments = [
        MultimodalSegment(
            segment_id="seg_0",
            start=10,
            end=20,
            transcript_text="This part discusses unrelated setup.",
        ),
        MultimodalSegment(
            segment_id="seg_1",
            start=100,
            end=130,
            transcript_text="The dashboard chart explains the memory retrieval failure.",
        ),
    ]
    metadata = VideoMetadata(video_id="v1", title="Video", duration=300)
    args = build_refinement_tool_call("memory retrieval failure").arguments

    ranges, evidence = resolve_refinement_ranges(
        video_id="v1",
        question="memory retrieval failure",
        segments=segments,
        settings=Settings(_env_file=None, CHROMA_PATH=tmp_path / "chroma"),
        metadata=metadata,
        arguments=args,
    )

    assert evidence
    assert evidence[0].segment_id == "seg_1"
    assert ranges[0].start <= 100
    assert ranges[0].end >= 130
    assert "semantic_text_match" in ranges[0].reason


def test_refinement_tool_prefers_agent_explicit_ranges(tmp_path: Path) -> None:
    explicit = [RefineRange(start=50, end=70, reason="agent_selected", priority=1)]
    call = build_refinement_tool_call("question", target_ranges=explicit)

    ranges, evidence = resolve_refinement_ranges(
        video_id="v1",
        question="question",
        segments=[
            MultimodalSegment(segment_id="seg_0", start=100, end=130, transcript_text="question text")
        ],
        settings=Settings(_env_file=None, CHROMA_PATH=tmp_path / "chroma"),
        metadata=VideoMetadata(video_id="v1", title="Video", duration=300),
        arguments=call.arguments,
    )

    assert evidence == []
    assert ranges == explicit


def test_execute_refinement_tool_calls_partial_restart_with_resolved_ranges(tmp_path: Path, monkeypatch) -> None:
    from app import pipeline

    called = {}

    def fake_partial_restart(video_id, target_ranges, question, settings):
        called["video_id"] = video_id
        called["ranges"] = target_ranges
        called["question"] = question
        return EvidenceRefinementToolResult(
            status="updated",
            resolved_ranges=target_ranges,
            added_frames=2,
            added_captions=2,
        )

    monkeypatch.setattr(pipeline, "partial_restart_for_evidence", fake_partial_restart)
    (tmp_path / "metadata.json").write_text(
        VideoMetadata(video_id="v1", title="Video", duration=300).model_dump_json(),
        encoding="utf-8",
    )
    call = EvidenceRefinementToolCall(
        tool="partial_restart_for_evidence",
        arguments={"target_ranges": [{"start": 20, "end": 40, "reason": "agent", "priority": 1}]},
    )

    result = execute_refinement_tool(
        video_id="v1",
        question="question",
        video_dir=tmp_path,
        segments=[MultimodalSegment(segment_id="seg_0", start=20, end=40, transcript_text="question")],
        tool_call=call,
        settings=Settings(_env_file=None, CHROMA_PATH=tmp_path / "chroma"),
    )

    assert result.status == "updated"
    assert result.added_frames == 2
    assert called["video_id"] == "v1"
    assert called["ranges"][0].start == 20
