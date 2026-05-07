from app.models import EvidenceStatus, StorylineNode
from app.reasoning.storyline import build_storyline_from_segments
from app.transcript.segmenter import segment_transcript


def test_storyline_node_without_evidence_is_not_supported():
    node = StorylineNode(
        node_id="n1",
        time_start=0,
        time_end=1,
        topic="claim",
        claim="unsupported claim",
        evidence_segment_ids=[],
        uncertainty=0.8,
        status=EvidenceStatus.supported,
    )
    assert node.status in {EvidenceStatus.unsupported, EvidenceStatus.weakly_supported}


def test_storyline_references_segments():
    segments = segment_transcript([{"start": 0, "end": 90, "text": "可追溯知识笔记需要 segment_id。"}])
    storyline = build_storyline_from_segments("v1", segments)
    assert storyline.nodes
    assert storyline.nodes[0].evidence_segment_ids == [segments[0].segment_id]
