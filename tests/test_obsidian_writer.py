from pathlib import Path

from app.memory.obsidian_writer import safe_filename, write_obsidian_notes
from app.modality.router import route_modality
from app.models import VideoMetadata
from app.reasoning.storyline import build_storyline_from_segments
from app.reasoning.summarizer import summarize_video
from app.transcript.segmenter import segment_transcript


def test_obsidian_writer_writes_legal_markdown(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    metadata = VideoMetadata(video_id="v1", title='Bad:/Title*?', source="mock", duration=120)
    segments = segment_transcript([{"start": 0, "end": 80, "text": "这是一个测试视频，讨论知识笔记和证据。"}])
    profile = route_modality(metadata, segments)
    summary = summarize_video(metadata, segments, profile)
    storyline = build_storyline_from_segments("v1", segments)
    paths = write_obsidian_notes(vault, metadata, summary, storyline)
    assert paths["video_note"].exists()
    assert paths["storyline_note"].exists()
    assert paths["evidence_note"].exists()
    assert paths["qa_note"].exists()
    assert paths["updates_note"].exists()
    text = paths["video_note"].read_text(encoding="utf-8")
    assert "# 30 秒速览" in text
    assert "video_id: v1" in text
    assert paths["video_note"].name == "index.md"
    assert paths["video_note"].parent.name == "v1 - Bad--Title"
    assert (vault / "40_MOCs" / "Video Index.md").exists()
    assert safe_filename(metadata.title) == "Bad--Title"
