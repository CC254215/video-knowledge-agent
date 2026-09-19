from app.models import CanonicalEntity, MultimodalSegment, SummaryReport, VideoMetadata
from app.reasoning.qa_router import route_question
from app.transcript.entity_normalizer import normalize_segments


def test_normalization_preserves_raw_and_only_replaces_named_aliases(tmp_path):
    segment = MultimodalSegment(segment_id="s1", start=0, end=5, transcript_text="打开 WalkBody，然后点击按钮")
    summary = SummaryReport(
        video_id="v1", quick_overview=[], structured_outline=[], deep_analysis=[], action_items=[],
        important_quotes=[], open_questions=[], modality_note="", canonical_entities=[
            CanonicalEntity(canonical="Workbuddy", aliases=["WalkBody"], entity_type="software", confidence=0.96)
        ],
    )
    updated, _ = normalize_segments(
        [segment], VideoMetadata(video_id="v1", title="Workbuddy Tutorial"), summary,
        output_path=tmp_path / "entity_normalization.json",
    )
    assert updated[0].raw_transcript_text == "打开 WalkBody，然后点击按钮"
    assert updated[0].normalized_transcript_text == "打开 Workbuddy，然后点击按钮"
    assert updated[0].transcript_text == updated[0].normalized_transcript_text
    assert updated[0].replacements[0]["canonical"] == "Workbuddy"


def test_low_confidence_entity_is_not_applied():
    segment = MultimodalSegment(segment_id="s1", start=0, end=5, transcript_text="打开 WalkBody")
    summary = SummaryReport(
        video_id="v1", quick_overview=[], structured_outline=[], deep_analysis=[], action_items=[],
        important_quotes=[], open_questions=[], modality_note="", canonical_entities=[
            CanonicalEntity(canonical="Workbuddy", aliases=["WalkBody"], confidence=0.6)
        ],
    )
    updated, _ = normalize_segments([segment], VideoMetadata(video_id="v1", title="Unrelated"), summary)
    assert updated[0].transcript_text == "打开 WalkBody"
    assert updated[0].raw_transcript_text == "打开 WalkBody"


def test_fast_router_is_conservative():
    assert route_question("老师使用了什么软件？").mode == "fast"
    assert route_question("老师为什么选择这个软件？").mode == "full"
    assert route_question("老师先安装软件还是先配置环境？").mode == "full"
    assert route_question("和上一个视频相比有什么不同？").mode == "full"
    assert route_question("老师点击的是哪个按钮？").mode == "full"
