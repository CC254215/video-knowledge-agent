from app.models import ConfidenceLevel, ConversationTurn, OutlineItem, SummaryReport
from app.ui.formatters import format_chat_answer_for_ui, format_evidence_for_ui, format_evidence_gallery_items, format_summary_for_ui


def test_chat_answer_formats_answer_evidence_timestamps_confidence():
    turn = ConversationTurn(
        turn_id="t1",
        video_id="v1",
        user_question="Q",
        answer="回答内容",
        evidence_ids=["seg_0000"],
        evidence=[
            {
                "evidence_id": "v1:seg_0000:speech",
                "segment_id": "seg_0000",
                "evidence_type": "speech",
                "text": "原始片段文本",
                "start": 10,
                "end": 20,
            }
        ],
        timestamps=["[00:00:10]"],
        evidence_types=["speech"],
        confidence=ConfidenceLevel.high,
        needs_visual_check=False,
        reason="证据充分",
    )

    answer_md = format_chat_answer_for_ui(turn)
    evidence_md = format_evidence_for_ui(turn)

    assert "回答内容" in answer_md
    assert "high" in answer_md
    assert "[00:00:10]" in evidence_md
    assert "speech" in evidence_md
    assert "原始片段文本" in evidence_md


def test_evidence_gallery_items_include_frame_path_and_caption():
    items = format_evidence_gallery_items(
        [
            {
                "image_path": "D:/video_knowledge_agent/data/videos/v1/frames/raw/frame_001.jpg",
                "timestamp": "[00:10:00]",
                "evidence_type": "frame_caption",
                "frame_caption": "画面展示交易曲线和风险控制说明。",
            }
        ]
    )

    assert items
    assert items[0][0].endswith("frame_001.jpg")
    assert "交易曲线" in items[0][1]


def test_summary_ui_shows_full_structured_outline():
    summary = SummaryReport(
        video_id="v1",
        quick_overview=[],
        structured_outline=[
            OutlineItem(timestamp=f"[00:{index:02d}:00]", topic=f"topic-{index}", key_points=["point"], segment_ids=[f"seg_{index:04d}"])
            for index in range(10)
        ],
        deep_analysis=[],
        action_items=[],
        important_quotes=[],
        open_questions=[],
        modality_note="test",
    )

    html = format_summary_for_ui(summary)

    assert "topic-0" in html
    assert "topic-9" in html
