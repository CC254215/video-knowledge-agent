from app.transcript.segmenter import segment_transcript


def test_segmenter_splits_long_transcript_with_timestamps():
    raw = [
        {"start": 0, "end": 40, "text": "第一段介绍 agent memory 和 evidence。"},
        {"start": 40, "end": 80, "text": "接下来讨论 segment_id 和 timestamp。"},
        {"start": 80, "end": 130, "text": "总结一下所有结论必须有证据。"},
    ]
    segments = segment_transcript(raw, min_seconds=30, max_seconds=70)
    assert len(segments) >= 2
    assert all(segment.start <= segment.end for segment in segments)
    assert all(segment.text for segment in segments)
    assert all(segment.segment_id.startswith("seg_") for segment in segments)
