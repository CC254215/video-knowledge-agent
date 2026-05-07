from pathlib import Path

from app.memory.obsidian_writer import render_video_note
from app.config import Settings
from app.models import ModalityMode, ModalityProfile, MultimodalSegment, Storyline, SummaryReport, VideoMetadata
from app.reasoning.conversation_agent import ConversationAgent
from app.reasoning.storyline import build_storyline_from_multimodal_segments
from app.reasoning.summarizer import summarize_multimodal_video
from app.transcript.multimodal_segmenter import save_multimodal_segments


def test_evidence_types_filter_out_ocr():
    segment = MultimodalSegment(segment_id="seg_0000", start=0, end=1, transcript_text="文本", evidence_types=["speech", "ocr"])  # type: ignore[list-item]
    assert segment.evidence_types == ["speech"]


def test_qa_answer_does_not_reference_ocr(tmp_path: Path):
    segment = MultimodalSegment(segment_id="seg_0000", start=0, end=1, transcript_text="文本证据")
    save_multimodal_segments([segment], tmp_path)
    turn = ConversationAgent("v1", tmp_path).answer("核心观点是什么？")
    assert "ocr" not in [item.lower() for item in turn.evidence_types]
    assert "OCR" not in turn.answer


def test_obsidian_says_ocr_disabled_not_evidence():
    metadata = VideoMetadata(video_id="v1", title="测试")
    segment = MultimodalSegment(segment_id="seg_0000", start=0, end=1, transcript_text="文本证据")
    profile = ModalityProfile(mode=ModalityMode.text_dominant, speech_density=1.0, visual_required_score=0.0, ocr_required_score=0.0, reason="text")
    summary = summarize_multimodal_video(metadata, [segment], profile, settings=Settings(_env_file=None, OPENAI_API_KEY=None))
    storyline = build_storyline_from_multimodal_segments("v1", [segment], settings=Settings(_env_file=None, OPENAI_API_KEY=None))
    text = render_video_note(metadata, summary, storyline, multimodal_segments=[segment])
    assert "OCR" in text
    assert "已禁用" in text or "disabled" in text
    assert "evidence_type=ocr" not in text.lower()
