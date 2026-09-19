from app.models import ConfidenceLevel, ConversationTurn, EvidenceStatus, OutlineItem, Storyline, StorylineNode, SummaryReport
from types import SimpleNamespace

from app.ui.formatters import (
    format_chat_answer_for_ui,
    format_evidence_for_ui,
    format_evidence_gallery_items,
    format_processing_timeline,
    format_storyline_for_ui,
    format_status_for_ui,
    format_summary_for_ui,
)


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


def test_status_distinguishes_video_duration_from_processing_time():
    result = SimpleNamespace(
        success=True,
        video_id="v1",
        metadata=SimpleNamespace(title="Video", author="Author", duration=935.8),
        modality_profile=None,
        obsidian_status="",
        processing_state={
            "stages": {
                "pipeline": {"status": "succeeded", "duration_seconds": 218.3, "total_duration_seconds": 1556.8},
                "summary": {"status": "succeeded", "duration_seconds": 109.0},
            }
        },
    )
    html = format_status_for_ui(result)
    assert "视频时长" in html
    assert "15分36秒" in html
    assert "处理耗时" in html
    assert "总计 25分57秒" in html
    assert "生成结构化摘要" in html


def test_timing_visualization_shows_granular_ingest_stages():
    html = format_processing_timeline(
        {
            "stages": {
                "pipeline": {"status": "succeeded", "total_duration_seconds": 20},
                "video_download": {"status": "succeeded", "duration_seconds": 7},
                "asr": {"status": "succeeded", "duration_seconds": 10},
            }
        }
    )
    assert "下载视频" in html
    assert "语音识别" in html
    assert "timing-fill succeeded" in html


def test_timing_total_includes_stages_retained_across_resume():
    html = format_processing_timeline(
        {
            "stages": {
                "pipeline": {"status": "succeeded", "total_duration_seconds": 90},
                "asr": {"status": "succeeded", "duration_seconds": 290},
                "summary": {"status": "succeeded", "duration_seconds": 50},
            }
        }
    )
    assert "总计 5分40秒" in html


def test_storyline_card_displays_coverage_range_and_evidence_count():
    storyline = Storyline(
        video_id="v1",
        query="q",
        nodes=[
            StorylineNode(
                node_id="n1",
                time_start=120,
                time_end=300,
                topic="topic",
                claim="claim",
                evidence_segment_ids=["s1", "s2"],
                confidence=0.9,
                uncertainty=0.1,
                status=EvidenceStatus.supported,
            )
        ],
    )
    html = format_storyline_for_ui(storyline)
    assert "[00:02:00] - [00:05:00]" in html
    assert "证据片段 2 个" in html
