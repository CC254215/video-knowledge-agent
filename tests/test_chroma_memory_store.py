from app.config import Settings
from app.models import FrameCaption, MultimodalSegment
from app.retrieval.chroma_memory import ChromaMemoryStore


def test_chroma_memory_store_indexes_speech_and_frame_caption():
    store = ChromaMemoryStore(Settings(_env_file=None))
    segments = [
        MultimodalSegment(
            segment_id="seg_0000",
            start=0,
            end=10,
            transcript_text="作者认为证据必须绑定时间戳。",
            representative_frame_ids=["f0"],
            visual_captions=[FrameCaption(frame_id="f0", timestamp=2, caption="画面是一个标题页。", model="mock")],
        )
    ]
    store.create_video_collection("v1")
    store.add_multimodal_segments("v1", segments)

    results = store.search("v1", "时间戳证据", top_k=5)
    assert results
    assert {item.evidence_type for item in results} >= {"speech"}
    assert all(item.segment_id == "seg_0000" for item in results)

    visual = store.search("v1", "标题页", top_k=5, evidence_types=["frame_caption"])
    assert visual
    assert visual[0].evidence_type == "frame_caption"
    assert visual[0].frame_id == "f0"
