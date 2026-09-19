from app.models import MultimodalSegment, RetrievedEvidence
from app.reasoning.conversation_agent import _global_coverage_bins


def test_global_coverage_uses_observation_timestamp_not_segment_span() -> None:
    segments = [MultimodalSegment(segment_id="seg_0000", start=0, end=180)]
    evidence = [
        RetrievedEvidence(
            evidence_id="frame@80",
            video_id="v1",
            segment_id="seg_0000",
            evidence_type="frame_caption",
            text="a visible frame",
            start=80,
            end=80,
            timestamp=80,
        )
    ]
    assert _global_coverage_bins(evidence, segments, ["visual"]) == 1
