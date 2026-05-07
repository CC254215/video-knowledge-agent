from pathlib import Path

from app.config import Settings
from app.models import FrameCaption, TranscriptSegment, VideoFrame
from app.reasoning.conversation_agent import ConversationAgent
from app.transcript.multimodal_segmenter import build_multimodal_segments, save_multimodal_segments


def test_chat_returns_time_ordered_text_and_frame_evidence(tmp_path: Path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake")
    segments = build_multimodal_segments(
        [TranscriptSegment(segment_id="seg_0000", start=0, end=10, text="作者解释气球实验中小气球的气会流向大气球。")],
        [VideoFrame(frame_id="f0", timestamp=2, path=str(image), selected=True)],
        {"seg_0000": ["f0"]},
        [FrameCaption(frame_id="f0", timestamp=2, image_path=str(image), caption="画面展示一大一小两个气球通过管子相连。", model="fake", status="success")],
    )
    save_multimodal_segments(segments, tmp_path)

    turn = ConversationAgent("v1", tmp_path, settings=Settings(_env_file=None, CHROMA_PATH=tmp_path / "chroma")).answer("气球实验里发生了什么？")

    assert turn.evidence
    assert turn.timestamps
    first = turn.evidence[0]
    assert "time_start" in first and "time_end" in first
    assert first["text"]
    assert any(item.get("frame_caption") for item in turn.evidence)
    assert any(item.get("image_path") == str(image) for item in turn.evidence)
    assert "ocr" not in turn.evidence_types
