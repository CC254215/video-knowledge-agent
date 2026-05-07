from __future__ import annotations

import json
from pathlib import Path

from app.analysis.multi_video_aggregator import MultiVideoAggregator
from app.config import Settings
from app.models import EvidenceStatus, MultimodalSegment, Storyline, StorylineNode, VideoMetadata
from app.transcript.multimodal_segmenter import save_multimodal_segments


def _write_video(root: Path, video_id: str, title: str, author: str, claim: str, segment_text: str) -> None:
    video_dir = root / "videos" / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    metadata = VideoMetadata(video_id=video_id, title=title, author=author, duration=120)
    (video_dir / "metadata.json").write_text(metadata.model_dump_json(), encoding="utf-8")
    save_multimodal_segments(
        [
            MultimodalSegment(
                segment_id="seg_0000",
                start=0,
                end=60,
                transcript_text=segment_text,
                evidence_types=["speech"],
            )
        ],
        video_dir,
    )
    storyline = Storyline(
        video_id=video_id,
        query="聚合测试",
        nodes=[
            StorylineNode(
                node_id="node_0000",
                time_start=0,
                time_end=60,
                topic="Agent memory",
                claim=claim,
                title="Agent memory pipeline",
                summary=claim,
                key_points=[claim],
                evidence_segment_ids=["seg_0000"],
                speech_evidence_ids=["seg_0000"],
                uncertainty=0.1,
                status=EvidenceStatus.supported,
            )
        ],
    )
    (video_dir / "storyline.json").write_text(storyline.model_dump_json(), encoding="utf-8")


def test_multi_video_aggregator_exports_topics(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    vault = tmp_path / "vault"
    vault.mkdir()
    _write_video(
        data_dir,
        "v1",
        "Agent Memory A",
        "author-a",
        "Agent memory should combine retrieval and reflection before writing long term notes.",
        "The speaker explains retrieval, reflection, and long term memory notes.",
    )
    _write_video(
        data_dir,
        "v2",
        "Agent Memory B",
        "author-a",
        "Agent memory benefits from evidence grounded retrieval before durable memory consolidation.",
        "The speaker compares retrieval evidence and durable memory consolidation.",
    )
    settings = Settings(data_dir=data_dir, chroma_path=tmp_path / "chroma", obsidian_vault_path=vault)

    result = MultiVideoAggregator(settings=settings).run(k=1, export_obsidian=True)

    assert result["record_count"] == 2
    assert result["topic_count"] == 1
    assert result["memory_saved"] == 1
    assert result["obsidian_paths"]
    note = Path(result["obsidian_paths"][0])
    assert note.exists()
    text = note.read_text(encoding="utf-8")
    assert "## 视频与节点" in text
    assert "## 共识观点" in text
    assert "cluster_id: 0" in text
    assert json.dumps(result, ensure_ascii=False)
