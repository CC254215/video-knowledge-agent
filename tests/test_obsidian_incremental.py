from pathlib import Path

from app.memory.obsidian_writer import write_obsidian_notes
from app.models import (
    ConfidenceLevel,
    ConversationTurn,
    FrameCaption,
    MultimodalSegment,
    Storyline,
    SummaryReport,
    VideoMetadata,
)


def _summary() -> SummaryReport:
    return SummaryReport(
        video_id="v1",
        quick_overview=["overview"],
        structured_outline=[],
        deep_analysis=[],
        action_items=[],
        important_quotes=[],
        open_questions=[],
        modality_note="text",
    )


def test_obsidian_incremental_updates_do_not_duplicate_main_note(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    metadata = VideoMetadata(video_id="v1", title="Same Video")
    storyline = Storyline(video_id="v1", query="q", nodes=[])
    segment = MultimodalSegment(
        segment_id="seg_0000",
        start=0,
        end=10,
        transcript_text="speech",
        representative_frame_ids=["f0"],
        visual_captions=[FrameCaption(frame_id="f0", timestamp=1, caption="frame caption", model="fake", status="success")],
    )
    turn = ConversationTurn(
        turn_id="t1",
        video_id="v1",
        user_question="Q",
        answer="A",
        evidence_ids=["seg_0000"],
        confidence=ConfidenceLevel.high,
        needs_visual_check=False,
        reason="ok",
    )

    first = write_obsidian_notes(vault, metadata, _summary(), storyline, multimodal_segments=[segment], conversation_history=[turn])
    second = write_obsidian_notes(vault, metadata, _summary(), storyline, multimodal_segments=[segment], conversation_history=[turn])

    notes = list((vault / "10_Sources" / "Videos").glob("*/index.md"))
    assert len(notes) == 1
    assert first["video_note"] == second["video_note"]
    text = notes[0].read_text(encoding="utf-8")
    assert text.count("type: video-note") == 1
    updates = first["updates_note"].read_text(encoding="utf-8")
    assert updates.count("audit_trace_id") >= 2
    assert first["storyline_note"] == second["storyline_note"]
