from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.evidence.orchestrator import EvidenceOrchestrationResult, EvidenceOrchestrator
from app.models import (
    ConfidenceLevel,
    ConversationTurn,
    EvidencePlan,
    FrameCaption,
    MultimodalSegment,
    QuestionAnalysis,
    RetrievedEvidence,
    VideoFrame,
    VideoMetadata,
)
from app.reasoning.conversation_agent import ConversationAgent, _validated_fact_support
from app.reasoning.evidence_gap_planner import EvidenceAcquisitionPlan
from app.reasoning.evidence_refinement_tool import (
    EvidenceRefinementToolArguments,
    EvidenceRefinementToolResult,
    build_refinement_tool_call,
    resolve_refinement_ranges,
)
from app.transcript.multimodal_segmenter import save_multimodal_segments
from app.vision.frame_refiner import FrameEvidenceRefiner, _temporal_stratified_select
from app.vision.correlation_agent import assess_video_text_correlation


def _turn(*, sufficient: bool, confidence: ConfidenceLevel = ConfidenceLevel.low) -> ConversationTurn:
    return ConversationTurn(
        turn_id="t1",
        video_id="v1",
        user_question="question",
        answer="answer" if sufficient else "当前视频证据不足以支持这个结论。",
        confidence=confidence,
        agreement_score=0.8 if sufficient else 0.0,
        needs_visual_check=not sufficient,
        evidence_sufficient=sufficient,
        reason="test",
    )


def test_persist_false_still_runs_refinement_when_allowed(monkeypatch, tmp_path: Path) -> None:
    save_multimodal_segments([MultimodalSegment(segment_id="s1", start=0, end=10, transcript_text="lesson")], tmp_path)
    called = {"count": 0}

    monkeypatch.setattr(ConversationAgent, "_answer_once", lambda *args, **kwargs: _turn(sufficient=False))

    def fake_review(self, question, turn, segments, evidence, metadata_loader, reanswer):
        called["count"] += 1
        turn.stop_reason = "planner_answer_with_existing_evidence"
        return EvidenceOrchestrationResult(turn=turn, segments=segments, handled=True)

    monkeypatch.setattr(EvidenceOrchestrator, "review_and_maybe_refine", fake_review)
    agent = ConversationAgent("v1", tmp_path, settings=Settings(_env_file=None, evidence_llm_api_key=None))

    agent.answer("question", persist=False, allow_refinement=True)

    assert called["count"] == 1
    assert not (tmp_path / "qa_history.json").exists()


def test_sufficient_speech_answer_does_not_refine(monkeypatch, tmp_path: Path) -> None:
    save_multimodal_segments([MultimodalSegment(segment_id="s1", start=0, end=10, transcript_text="dataset then optimizer")], tmp_path)
    turn = _turn(sufficient=True, confidence=ConfidenceLevel.high)
    turn.evidence = [{"evidence_id": "v1:s1:speech", "segment_id": "s1", "evidence_type": "speech", "text": "dataset then optimizer", "start": 0, "end": 10}]
    monkeypatch.setattr(ConversationAgent, "_answer_once", lambda *args, **kwargs: turn)
    monkeypatch.setattr(FrameEvidenceRefiner, "refine", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected VLM refinement")))

    result = ConversationAgent("v1", tmp_path, settings=Settings(_env_file=None, evidence_llm_api_key=None)).answer(
        "老师先介绍数据集还是先介绍优化器？", persist=False, allow_refinement=True
    )

    assert result.stop_reason == "initial_evidence_sufficient"


def test_silent_whole_video_uses_three_coverage_ranges(tmp_path: Path) -> None:
    args = EvidenceRefinementToolArguments(
        query="main stages",
        has_real_speech=False,
        temporal_scope="whole_video",
        temporal_relation="stage",
        requires_global_coverage=True,
    )
    ranges, evidence = resolve_refinement_ranges(
        "v1",
        "main stages",
        [MultimodalSegment(segment_id="s1", start=0, end=180)],
        Settings(_env_file=None, chroma_path=tmp_path / "chroma"),
        VideoMetadata(video_id="v1", title="silent", duration=180),
        args,
    )

    assert evidence == []
    assert [(row.start, row.end) for row in ranges] == [(0.0, 60.0), (60.0, 120.0), (120.0, 180.0)]


def test_decision_fact_requires_structured_direct_support() -> None:
    evidence = [RetrievedEvidence(
        evidence_id="v1:s1:frame_caption:f1", video_id="v1", segment_id="s1",
        evidence_type="frame_caption", text="a person is holding a tool", start=10, end=10, score=1.0,
    )]
    plan = EvidencePlan(decision_facts=["a wooden plank is used"])
    support, missing = _validated_fact_support(
        {
            "evidence_sufficient": False,
            "fact_support": [{
                "fact": "a wooden plank is used", "status": "unsupported",
                "evidence_ids": [], "reason": "tool does not identify a wooden plank",
            }],
        },
        plan,
        evidence,
    )

    assert support[0]["status"] == "unsupported"
    assert missing == ["decision_fact unsupported: a wooden plank is used"]


def test_orchestrator_stops_after_one_refinement(monkeypatch, tmp_path: Path) -> None:
    segments = [MultimodalSegment(segment_id="s1", start=0, end=10, transcript_text="lesson")]
    save_multimodal_segments(segments, tmp_path)
    initial = _turn(sufficient=False)
    initial.question_analysis = QuestionAnalysis(content_modalities=["visual"])
    initial.evidence_plan = EvidencePlan(required_modalities=["visual"])
    initial.structured_evidence_gate = {"satisfied_constraints": [], "unmet_constraints": ["required modality visual is unavailable"]}
    plan = EvidenceAcquisitionPlan(
        decision="acquire_more_evidence",
        evidence_sufficiency="insufficient",
        actions=[build_refinement_tool_call("question", target_ranges=[])],
    )
    monkeypatch.setattr("app.evidence.orchestrator.EvidenceGapPlanner.decide", lambda *args, **kwargs: plan)
    monkeypatch.setattr("app.evidence.orchestrator.execute_refinement_tool", lambda *args, **kwargs: EvidenceRefinementToolResult(status="updated", added_frames=1))
    calls = {"count": 0}

    def reanswer(updated, refined):
        calls["count"] += 1
        return _turn(sufficient=False)

    class Store:
        def add_multimodal_segments(self, *args, **kwargs):
            return None

    result = EvidenceOrchestrator("v1", tmp_path, Settings(_env_file=None), store=Store()).review_and_maybe_refine(
        "question", initial, segments, [], lambda: VideoMetadata(video_id="v1", title="video", duration=10), reanswer
    )

    assert calls["count"] == 1
    assert result.turn.stop_reason == "insufficient_after_single_refinement"


def test_multi_range_refinement_keeps_all_image_paths(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    (tmp_path / "metadata.json").write_text(
        VideoMetadata(video_id="v1", title="video", duration=30, local_path=str(source)).model_dump_json(), encoding="utf-8"
    )

    def fake_extract(video_id, video_path, video_dir, fps=1.0, start=None, end=None, frame_output_dir=None):
        out = Path(frame_output_dir)
        out.mkdir(parents=True, exist_ok=True)
        frames = []
        for index in range(3):
            path = out / f"frame_{index:06d}.jpg"
            path.write_bytes(f"{start}-{index}".encode())
            timestamp = float(start or 0) + index
            frames.append(VideoFrame(frame_id=f"f_{int(timestamp * 1000):010d}", timestamp=timestamp, path=str(path)))
        return frames

    monkeypatch.setattr("app.vision.frame_refiner.extract_frames", fake_extract)
    monkeypatch.setattr(
        "app.vision.frame_refiner.VLMCaptioner.caption_frames",
        lambda self, frames: [FrameCaption(frame_id=f.frame_id, timestamp=f.timestamp, image_path=f.path, caption="frame", model="test") for f in frames],
    )
    segments = [MultimodalSegment(segment_id="s1", start=0, end=30)]
    refiner = FrameEvidenceRefiner(tmp_path, settings=Settings(_env_file=None))
    segments = refiner.refine("v1", "question", (0, 10), segments, refinement_id="run1", range_index=0)
    segments = refiner.refine("v1", "question", (20, 30), segments, refinement_id="run1", range_index=1)

    paths = [Path(caption.image_path) for caption in segments[0].visual_captions if caption.image_path]
    assert paths
    assert all(path.exists() for path in paths)
    assert {path.parent.name for path in paths} == {"range_000", "range_001"}


def test_temporal_stratified_selection_spans_full_range() -> None:
    frames = [VideoFrame(frame_id=f"f{i}", timestamp=float(i), path=f"f{i}.jpg") for i in range(30)]
    selected = _temporal_stratified_select(frames, 3)

    assert selected[0].timestamp < 10
    assert 10 <= selected[1].timestamp < 20
    assert selected[2].timestamp >= 20


def test_silent_correlation_fallback_is_marked_unavailable(tmp_path: Path) -> None:
    profile = assess_video_text_correlation(
        "v1", [], video_path=None, output_dir=tmp_path, settings=Settings(_env_file=None)
    )

    assert profile.correlation_available is False
    assert profile.fallback_reason == "no_valid_transcript"
    assert profile.refine_frame_count == 10
