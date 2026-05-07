from pathlib import Path

from app.config import Settings
from app.models import FrameCaption, MultimodalSegment
from app.retrieval.chroma_memory import ChromaMemoryStore


def test_persistent_chroma_survives_store_restart(tmp_path: Path):
    settings = Settings(_env_file=None, CHROMA_PATH=tmp_path / "chroma")
    segment = MultimodalSegment(
        segment_id="seg_0000",
        start=0,
        end=10,
        transcript_text="视频讲解证据可追溯的知识 Agent。",
        representative_frame_ids=["f0"],
        visual_captions=[
            FrameCaption(
                frame_id="f0",
                timestamp=2,
                image_path=str(tmp_path / "f0.jpg"),
                caption="画面展示 Video Knowledge Agent 的标题页。",
                model="fake",
                status="success",
            )
        ],
    )

    first = ChromaMemoryStore(settings)
    first.delete_video_collection("persist_v1")
    first.add_multimodal_segments("persist_v1", [segment])

    second = ChromaMemoryStore(settings)
    speech = second.search("persist_v1", "知识 Agent", evidence_types=["speech"])
    visual = second.search("persist_v1", "标题页", evidence_types=["frame_caption"])
    assert speech
    assert visual
    assert visual[0].frame_id == "f0"
