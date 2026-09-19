from __future__ import annotations

from app.config import Settings
from app.models import (
    ActionItem,
    AnalysisItem,
    ConfidenceLevel,
    EvidenceStatus,
    FrameCaption,
    MultimodalSegment,
    OutlineItem,
    QuoteItem,
    Storyline,
    StorylineNode,
    SummaryReport,
    VideoMetadata,
)
from app import pipeline


class RecordingAdapter:
    def __init__(self) -> None:
        self.raw = []
        self.summary = []

    def save_raw_evidence(self, payload):
        self.raw.append(payload)

    def save_model_summary(self, payload):
        self.summary.append(payload)

    def save_user_verified_insight(self, payload):
        pass

    def search_related_memories(self, query: str, limit: int = 5):
        return []

    def close(self):
        pass


def test_pipeline_saves_mempalace_summary_and_raw_evidence(monkeypatch, tmp_path) -> None:
    adapter = RecordingAdapter()
    monkeypatch.setattr(pipeline, "create_memory_adapter", lambda settings: adapter)
    settings = Settings(_env_file=None, data_dir=tmp_path, mempalace_provider="mcp_stdio")
    metadata = VideoMetadata(video_id="v1", title="Video A", author="author")
    report = SummaryReport(
        video_id="v1",
        quick_overview=["overview"],
        structured_outline=[OutlineItem(timestamp="00:00:00", topic="topic", key_points=["point"], segment_ids=["seg_0"])],
        deep_analysis=[AnalysisItem(claim="claim", evidence_segment_ids=["seg_0"])],
        action_items=[ActionItem(action="act", evidence_segment_ids=["seg_0"])],
        important_quotes=[QuoteItem(quote="quote", segment_id="seg_0", timestamp="00:00:00")],
        open_questions=["question"],
        modality_note="text",
        confidence=ConfidenceLevel.medium,
        generation_status="llm_success",
    )
    storyline = Storyline(
        video_id="v1",
        query="query",
        nodes=[
            StorylineNode(
                node_id="node_0",
                time_start=0,
                time_end=60,
                topic="topic",
                claim="claim",
                evidence_segment_ids=["seg_0"],
                uncertainty=0.1,
                status=EvidenceStatus.supported,
            )
        ],
    )
    segments = [
        MultimodalSegment(
            segment_id="seg_0",
            start=0,
            end=60,
            transcript_text="speech evidence",
            visual_captions=[FrameCaption(frame_id="f0", timestamp=10, caption="frame caption", model="test")],
        )
    ]

    stats = pipeline._save_mempalace_memories(settings, metadata, report, storyline, segments)

    assert stats["status"] == "saved"
    assert stats["model_summary"] == 1
    assert stats["raw_evidence"] == 2
    assert adapter.summary[0]["summary_type"] == "model_summary"
    assert {item["evidence_type"] for item in adapter.raw} == {"speech", "frame_caption"}
