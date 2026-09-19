from __future__ import annotations

import json
import uuid
from pathlib import Path

from app.config import Settings, get_settings
from app.models import FrameCaption, MultimodalSegment, VideoFrame, VideoMetadata
from app.models import VideoCorrelationProfile
from app.modality.router import question_needs_visual_check
from app.vision.frame_deduper import dedupe_frames
from app.vision.frame_extractor import extract_frames
from app.vision.vlm_captioner import VLMCaptioner, save_frame_captions
from app.transcript.multimodal_segmenter import save_multimodal_segments

VISUAL_TRIGGER_PHRASES = ("如图所示", "这里可以看到", "这个按钮", "这个图", "图中", "界面", "公式", "代码")


def should_refine_frames(
    question: str,
    confidence: str | None = None,
    agreement: float | None = None,
    transcript_text: str = "",
) -> bool:
    if confidence == "low":
        return True
    if agreement is not None and agreement < 0.3:
        return True
    if question_needs_visual_check(question):
        return True
    return any(phrase in transcript_text for phrase in VISUAL_TRIGGER_PHRASES)


class FrameEvidenceRefiner:
    def __init__(self, video_dir: str | Path, settings: Settings | None = None) -> None:
        self.video_dir = Path(video_dir)
        self.settings = settings or get_settings()

    def refine(
        self,
        video_id: str,
        question: str,
        target_time_range: tuple[float, float],
        segments: list[MultimodalSegment],
        max_new_frames: int | None = None,
        refinement_strategy: str = "local_first",
        refinement_id: str | None = None,
        range_index: int = 0,
        persist: bool = True,
        metadata: VideoMetadata | None = None,
    ) -> list[MultimodalSegment]:
        """Add frame and frame_caption evidence only. OCR is disabled for this stage."""
        start, end = target_time_range
        max_new_frames = max_new_frames or _default_refine_count(self.video_dir)
        added_frames = self._extract_refined_frames(
            video_id,
            question,
            start,
            end,
            max_new_frames,
            refinement_strategy=refinement_strategy,
            refinement_id=refinement_id,
            range_index=range_index,
            metadata=metadata,
        )
        captions = VLMCaptioner(self.settings).caption_frames(added_frames)
        if persist:
            save_frame_captions(_load_existing_captions(self.video_dir) + captions, self.video_dir)

        for segment in segments:
            if segment.end < start or segment.start > end:
                continue
            matching = [frame for frame in added_frames if segment.start <= frame.timestamp <= segment.end]
            matching_ids = {frame.frame_id for frame in matching}
            segment.frame_ids.extend(frame.frame_id for frame in matching if frame.frame_id not in segment.frame_ids)
            segment.representative_frame_ids.extend(
                frame.frame_id for frame in matching if frame.frame_id not in segment.representative_frame_ids
            )
            new_captions = [caption for caption in captions if caption.frame_id in matching_ids]
            segment.visual_captions.extend(new_captions)
            caption_text = " ".join(caption.caption for caption in new_captions if caption.caption)
            if caption_text:
                segment.visual_summary = " ".join(part for part in [segment.visual_summary, caption_text] if part)
            if matching and "frame" not in segment.evidence_types:
                segment.evidence_types.append("frame")
            if caption_text and "frame_caption" not in segment.evidence_types:
                segment.evidence_types.append("frame_caption")
            segment.ocr_text = ""
            segment.modality_weight = _weight(segment.transcript_text, segment.visual_summary, segment.representative_frame_ids)

        if persist:
            save_multimodal_segments(segments, self.video_dir)
        return segments

    def _extract_refined_frames(
        self,
        video_id: str,
        question: str,
        start: float,
        end: float,
        max_new_frames: int,
        refinement_strategy: str = "local_first",
        refinement_id: str | None = None,
        range_index: int = 0,
        metadata: VideoMetadata | None = None,
    ) -> list[VideoFrame]:
        metadata_path = self.video_dir / "metadata.json"
        if metadata is None and metadata_path.exists():
            metadata = VideoMetadata.model_validate(json.loads(metadata_path.read_text(encoding="utf-8")))
        if metadata is not None:
            source_path = metadata.local_path
            if source_path and Path(source_path).exists():
                try:
                    run_id = refinement_id or uuid.uuid4().hex
                    output_dir = self.video_dir / "frames" / "refined" / run_id / f"range_{range_index:03d}"
                    frames = extract_frames(
                        video_id,
                        source_path,
                        self.video_dir,
                        fps=2.0,
                        start=start,
                        end=end,
                        frame_output_dir=output_dir,
                    )
                    deduped = dedupe_frames(frames)
                    frames = (
                        _temporal_stratified_select(deduped, max_new_frames)
                        if refinement_strategy == "temporal_stratified"
                        else deduped[:max_new_frames]
                    )
                    for frame in frames:
                        frame.selected = True
                        frame.selection_reason = f"frame_refiner:{question[:80]}"
                    return frames
                except Exception:
                    pass
        refined_dir = self.video_dir / "frames" / "refined"
        refined_dir.mkdir(parents=True, exist_ok=True)
        fallback = []
        for index, ts in enumerate(_refined_timestamps(start, end)[:max_new_frames]):
            path = refined_dir / f"refined_{int(ts * 1000)}.jpg"
            fallback.append(
                VideoFrame(
                    frame_id=f"refined_{int(start * 1000)}_{index}",
                    timestamp=ts,
                    path=str(path),
                    selected=True,
                    selection_reason=f"frame_refiner:{question[:80]}",
                )
            )
        return fallback


def _refined_timestamps(start: float, end: float) -> list[float]:
    if end <= start:
        return [start]
    count = min(10, max(3, int(end - start) + 1))
    step = (end - start) / max(1, count - 1)
    return [start + step * index for index in range(count)]


def _temporal_stratified_select(frames: list[VideoFrame], max_count: int) -> list[VideoFrame]:
    if max_count <= 0 or len(frames) <= max_count:
        return frames[:max_count]
    selected: list[VideoFrame] = []
    for index in range(max_count):
        left = round(index * len(frames) / max_count)
        right = round((index + 1) * len(frames) / max_count)
        bucket = frames[left:max(left + 1, right)]
        selected.append(bucket[len(bucket) // 2])
    return selected


def _load_existing_captions(video_dir: Path) -> list[FrameCaption]:
    path = video_dir / "frame_captions.json"
    if not path.exists():
        return []
    return [FrameCaption.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def _default_refine_count(video_dir: Path) -> int:
    path = video_dir / "video_correlation.json"
    if path.exists():
        try:
            return max(1, VideoCorrelationProfile.model_validate(json.loads(path.read_text(encoding="utf-8"))).refine_frame_count)
        except Exception:
            pass
    return 10


def _weight(text: str, visual_summary: str, frame_ids: list[str]) -> dict[str, float]:
    text_score = min(1.0, len(text.strip()) / 220) if text.strip() else 0.0
    vision_score = 0.15 if frame_ids else 0.0
    if visual_summary.strip():
        vision_score = max(vision_score, min(0.8, len(visual_summary.strip()) / 180))
    total = text_score + vision_score
    if total <= 0:
        return {"text": 0.0, "vision": 0.0}
    return {"text": round(text_score / total, 3), "vision": round(vision_score / total, 3)}
