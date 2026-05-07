from __future__ import annotations

import json
from pathlib import Path

from app import pipeline
from app.config import Settings
from app.models import MultimodalSegment, RefineRange, TranscriptSegment, VideoMetadata, VideoTranscript
from app.reasoning.conversation_agent import _time_filtered_evidence
from app.transcript.multimodal_segmenter import save_multimodal_segments


def test_partial_restart_adds_refined_text_segment_for_target_range(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(data_dir=tmp_path / "data", chroma_path=tmp_path / "chroma")
    video_id = "v1"
    out_dir = pipeline.video_dir(video_id, settings)
    metadata = VideoMetadata(video_id=video_id, title="交易课程", duration=900)
    pipeline.write_json(out_dir / "metadata.json", metadata)
    transcript = VideoTranscript(
        video_id=video_id,
        segments=[
            TranscriptSegment(segment_id="asr_0", start=590, end=625, text="十分钟附近作者讲了仓位管理和风险控制。"),
        ],
        source="asr",
    )
    pipeline.write_json(out_dir / "transcript.json", transcript)
    save_multimodal_segments([MultimodalSegment(segment_id="seg_0", start=0, end=120, transcript_text="开头内容")], out_dir)
    monkeypatch.setattr(pipeline.FrameEvidenceRefiner, "refine", lambda self, video_id, question, target_time_range, segments: segments)

    result = pipeline.partial_restart_for_evidence(
        video_id,
        [RefineRange(start=540, end=660, reason="time_anchor")],
        "十分钟的时候作者做了什么",
        settings,
    )

    assert result.added_text_segments == 1
    rows = json.loads((out_dir / "multimodal_segments.json").read_text(encoding="utf-8"))
    assert any(row["segment_id"].startswith("refined_text_") and "仓位管理" in row["transcript_text"] for row in rows)


def test_time_filtered_evidence_prefers_target_time_segment() -> None:
    segments = [
        MultimodalSegment(segment_id="seg_old", start=500, end=520, transcript_text="八分多钟的内容"),
        MultimodalSegment(segment_id="refined_text_540_660", start=590, end=625, transcript_text="十分钟附近作者讲了仓位管理。"),
        MultimodalSegment(segment_id="seg_late", start=750, end=780, transcript_text="十二分钟之后的内容"),
    ]

    evidence = _time_filtered_evidence("v1", "十分钟的时候作者做了什么", segments, ["speech"])

    assert evidence
    assert evidence[0].segment_id == "refined_text_540_660"
    assert "仓位管理" in evidence[0].text
