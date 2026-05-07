from pathlib import Path

from app.config import Settings
from app.models import FrameCaption, TranscriptSegment, VideoFrame
from app.reasoning.conversation_agent import ConversationAgent
from app.reasoning.llm_client import LLMClient
from app.transcript.multimodal_segmenter import build_multimodal_segments, save_multimodal_segments
from app.vision.frame_deduper import dedupe_frames
from app.vision.frame_extractor import frames_from_directory
from app.vision.frame_refiner import FrameEvidenceRefiner, should_refine_frames
from app.vision.keyframe_selector import select_representative_frames
from app.vision.vlm_captioner import VLMCaptioner


def test_one_fps_frames_are_candidates_not_all_vlm_calls(tmp_path: Path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for index in range(3):
        (frame_dir / f"frame_{index:06d}.jpg").write_bytes(f"frame-{index}".encode())
    frames = frames_from_directory(frame_dir, fps=1.0, video_id="v1")
    selected = [frames[0]]
    captions = VLMCaptioner(settings=Settings(_env_file=None)).caption_frames(selected)
    assert len(frames) == 3
    assert len(captions) == 1


def test_frame_deduper_removes_duplicate_frames(tmp_path: Path):
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    third = tmp_path / "c.jpg"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    third.write_bytes(b"different")
    frames = [
        VideoFrame(frame_id="f1", timestamp=0, path=str(first)),
        VideoFrame(frame_id="f2", timestamp=1, path=str(second)),
        VideoFrame(frame_id="f3", timestamp=2, path=str(third)),
    ]
    deduped = dedupe_frames(frames)
    assert [frame.frame_id for frame in deduped] == ["f1", "f3"]


def test_multimodal_segment_aligns_transcript_and_frames_without_ocr():
    transcript = [TranscriptSegment(segment_id="seg_0000", start=0, end=10, text="这里可以看到图表。")]
    frames = [
        VideoFrame(frame_id="f0", timestamp=2, path="f0.jpg", selected=True),
        VideoFrame(frame_id="f1", timestamp=12, path="f1.jpg"),
    ]
    captions = [FrameCaption(frame_id="f0", timestamp=2, caption="屏幕上是一张趋势图。", model="mock")]
    segments = build_multimodal_segments(transcript, frames, {"seg_0000": ["f0"]}, captions, ocr_results=None)
    assert segments[0].frame_ids == ["f0"]
    assert segments[0].representative_frame_ids == ["f0"]
    assert segments[0].ocr_text == ""
    assert set(segments[0].evidence_types) == {"speech", "frame", "frame_caption"}
    assert set(segments[0].modality_weight) == {"text", "vision"}


def test_short_asr_segments_are_merged_into_multimodal_segment():
    transcript = [
        TranscriptSegment(segment_id="asr_0", start=0, end=8, text="第一句。"),
        TranscriptSegment(segment_id="asr_1", start=8.5, end=18, text="第二句继续同一话题。"),
        TranscriptSegment(segment_id="asr_2", start=18.5, end=30, text="第三句仍然继续。"),
    ]
    frames = [VideoFrame(frame_id="f0", timestamp=10, path="f0.jpg")]
    segments = build_multimodal_segments(transcript, frames, min_seconds=60, max_seconds=120)
    assert len(segments) == 1
    assert "第一句" in segments[0].transcript_text
    assert segments[0].representative_frame_ids == ["f0"]


def test_silence_gap_splits_multimodal_segments():
    transcript = [
        TranscriptSegment(segment_id="asr_0", start=0, end=5, text="第一段。"),
        TranscriptSegment(segment_id="asr_1", start=12, end=18, text="长静音后第二段。"),
    ]
    segments = build_multimodal_segments(transcript, [], silence_gap=4)
    assert len(segments) == 2


def test_conversation_agent_answer_contains_evidence_types_and_timestamps(tmp_path: Path):
    segment = build_multimodal_segments(
        [TranscriptSegment(segment_id="seg_0000", start=0, end=10, text="核心论点是证据必须绑定时间戳。")],
        [VideoFrame(frame_id="f0", timestamp=1, path="f0.jpg", selected=True)],
        {"seg_0000": ["f0"]},
        [FrameCaption(frame_id="f0", timestamp=1, caption="", model="mock")],
        [],
    )
    save_multimodal_segments(segment, tmp_path)
    turn = ConversationAgent("v1", tmp_path, settings=Settings(_env_file=None)).answer("核心论点是什么？")
    assert turn.evidence_types
    assert "ocr" not in turn.evidence_types
    assert turn.timestamps
    assert turn.confidence.value in {"high", "medium", "low"}


def test_conversation_agent_uses_llm_json_and_filters_evidence_ids(tmp_path: Path, monkeypatch):
    segment = build_multimodal_segments(
        [TranscriptSegment(segment_id="seg_0000", start=0, end=10, text="视频明确说核心论点是证据必须绑定时间戳。")],
        [VideoFrame(frame_id="f0", timestamp=1, path="f0.jpg", selected=True)],
        {"seg_0000": ["f0"]},
        [FrameCaption(frame_id="f0", timestamp=1, caption="", model="mock")],
        [],
    )
    save_multimodal_segments(segment, tmp_path)

    def fake_generate_json(self, prompt, schema_hint=None):
        assert "OCR 已禁用" in prompt
        return {
            "answer": "核心论点是证据必须绑定时间戳。[00:00:00]",
            "evidence_ids": ["seg_0000", "hallucinated_seg"],
            "confidence": "high",
            "reason": "证据来自 seg_0000。",
            "suggested_followup_questions": ["证据在哪里？"],
        }

    monkeypatch.setattr(LLMClient, "generate_json", fake_generate_json)
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="test",
        OPENAI_BASE_URL="https://example.test/v1",
        LLM_MODEL="test-model",
    )
    turn = ConversationAgent("v1", tmp_path, settings=settings).answer("核心论点是什么？")
    assert turn.answer.startswith("核心论点")
    assert turn.evidence_ids == ["seg_0000"]
    assert "hallucinated_seg" not in turn.evidence_ids


def test_visual_question_sets_needs_visual_check(tmp_path: Path):
    segment = build_multimodal_segments(
        [TranscriptSegment(segment_id="seg_0000", start=0, end=10, text="这里可以看到一个按钮。")],
        [VideoFrame(frame_id="f0", timestamp=1, path="f0.jpg", selected=True)],
        {"seg_0000": ["f0"]},
        [],
        [],
    )
    save_multimodal_segments(segment, tmp_path)
    turn = ConversationAgent("v1", tmp_path, settings=Settings(_env_file=None)).answer("画面中的按钮是什么？")
    assert turn.needs_visual_check is True
    assert "证据不足" in turn.answer


def test_low_agreement_can_trigger_frame_refiner_without_ocr(tmp_path: Path):
    segment = build_multimodal_segments(
        [TranscriptSegment(segment_id="seg_0000", start=0, end=10, text="这里可以看到图表。")],
        [],
    )
    refined = FrameEvidenceRefiner(tmp_path, settings=Settings(_env_file=None)).refine("v1", "图中是什么？", (0, 10), segment)
    assert refined[0].representative_frame_ids
    assert refined[0].ocr_text == ""
    assert "ocr" not in refined[0].evidence_types
    assert should_refine_frames("图中是什么？", confidence="low", agreement=0.1)


def test_insufficient_evidence_refuses_fabrication(tmp_path: Path):
    turn = ConversationAgent("missing", tmp_path, settings=Settings(_env_file=None)).answer("画面中的公式是什么？")
    assert "当前视频证据不足以支持这个结论" in turn.answer
    assert turn.evidence_ids == []
    assert turn.needs_visual_check is True


def test_keyframe_selector_picks_at_least_one_frame_per_segment():
    transcript = [TranscriptSegment(segment_id="seg_0000", start=0, end=10, text="demo")]
    frames = [VideoFrame(frame_id="f0", timestamp=1, path="f0.jpg"), VideoFrame(frame_id="f1", timestamp=5, path="f1.jpg")]
    selected = select_representative_frames(transcript, frames)
    assert selected["seg_0000"]
