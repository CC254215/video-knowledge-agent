from __future__ import annotations

import json
from pathlib import Path

from app.models import FrameCaption, MultimodalSegment, OCRResult, TranscriptSegment, VideoFrame
from app.transcript.segmenter import TOPIC_MARKERS, extract_entities, extract_keywords

DEFAULT_MIN_SECONDS = 60.0
DEFAULT_MAX_SECONDS = 120.0
DEFAULT_MAX_CHARS = 1000
DEFAULT_SILENCE_GAP = 4.0


def build_multimodal_segments(
    transcript_segments: list[TranscriptSegment],
    frames: list[VideoFrame],
    representative_frame_ids_by_segment: dict[str, list[str]] | None = None,
    captions: list[FrameCaption] | None = None,
    ocr_results: list[OCRResult] | None = None,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    max_chars: int = DEFAULT_MAX_CHARS,
    silence_gap: float = DEFAULT_SILENCE_GAP,
) -> list[MultimodalSegment]:
    """Merge ASR/subtitle chunks into multimodal event segments.

    OCR is intentionally ignored in this stage. The `ocr_text` field remains on the model for later
    re-enablement, but it is always written as an empty string and never participates in evidence.
    """
    if not transcript_segments:
        return []

    merged_transcripts = _merge_transcript_segments(
        transcript_segments,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        max_chars=max_chars,
        silence_gap=silence_gap,
    )
    caption_by_frame: dict[str, list[FrameCaption]] = {}
    for caption in captions or []:
        caption_by_frame.setdefault(caption.frame_id, []).append(caption)
    representative_frame_ids_by_segment = representative_frame_ids_by_segment or {}

    output: list[MultimodalSegment] = []
    for index, transcript in enumerate(merged_transcripts):
        segment_frames = [frame for frame in frames if transcript.start <= frame.timestamp <= transcript.end]
        representative_ids = _representative_ids_for_segment(
            transcript,
            segment_frames,
            representative_frame_ids_by_segment,
        )
        segment_captions = [caption for frame_id in representative_ids for caption in caption_by_frame.get(frame_id, [])]
        visual_summary = " ".join(caption.caption for caption in segment_captions if caption.caption).strip()
        combined = " ".join([transcript.text, visual_summary]).strip()
        output.append(
            MultimodalSegment(
                segment_id=f"seg_{index:04d}",
                start=transcript.start,
                end=transcript.end,
                transcript_text=transcript.text,
                frame_ids=[frame.frame_id for frame in segment_frames],
                representative_frame_ids=representative_ids,
                visual_captions=segment_captions,
                ocr_text="",
                visual_summary=visual_summary,
                modality_weight=_initial_modality_weight(transcript.text, visual_summary, representative_ids),
                keywords=extract_keywords(combined),
                entities=extract_entities(combined),
            )
        )
    return output


def save_multimodal_segments(segments: list[MultimodalSegment], video_dir: str | Path) -> Path:
    path = Path(video_dir) / "multimodal_segments.json"
    path.write_text(json.dumps([_segment_dump(segment) for segment in segments], ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_multimodal_segments(path: str | Path) -> list[MultimodalSegment]:
    target = Path(path)
    rows = json.loads(target.read_text(encoding="utf-8"))
    for item in rows:
        if "frame_captions" in item and "visual_captions" not in item:
            item["visual_captions"] = item["frame_captions"]
    return [MultimodalSegment.model_validate(item) for item in rows]


def _segment_dump(segment: MultimodalSegment) -> dict:
    data = segment.model_dump(mode="json")
    data["frame_captions"] = [caption.model_dump(mode="json") for caption in segment.visual_captions]
    return data


def _merge_transcript_segments(
    transcript_segments: list[TranscriptSegment],
    min_seconds: float,
    max_seconds: float,
    max_chars: int,
    silence_gap: float,
) -> list[TranscriptSegment]:
    buckets: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    for segment in sorted(transcript_segments, key=lambda item: item.start):
        if not current:
            current.append(segment)
            continue
        duration = segment.end - current[0].start
        char_count = len(" ".join(item.text for item in current)) + len(segment.text)
        gap = segment.start - current[-1].end
        marker_split = duration >= min_seconds and _has_topic_marker(segment.text)
        hard_split = duration >= max_seconds or char_count > max_chars or gap > silence_gap
        if marker_split or hard_split:
            buckets.append(current)
            current = [segment]
        else:
            current.append(segment)
    if current:
        buckets.append(current)

    merged: list[TranscriptSegment] = []
    for index, bucket in enumerate(buckets):
        text = " ".join(item.text.strip() for item in bucket if item.text.strip())
        merged.append(
            TranscriptSegment(
                segment_id=f"seg_{index:04d}",
                start=bucket[0].start,
                end=max(item.end for item in bucket),
                text=text,
                keywords=extract_keywords(text),
                entities=extract_entities(text),
            )
        )
    return merged


def _representative_ids_for_segment(
    transcript: TranscriptSegment,
    segment_frames: list[VideoFrame],
    representative_frame_ids_by_segment: dict[str, list[str]],
) -> list[str]:
    explicit = representative_frame_ids_by_segment.get(transcript.segment_id)
    if explicit is not None:
        return [frame_id for frame_id in explicit if any(frame.frame_id == frame_id for frame in segment_frames)]
    selected = [frame.frame_id for frame in segment_frames if frame.selected]
    if selected:
        return selected[:3]
    if not segment_frames:
        return []
    midpoint = (transcript.start + transcript.end) / 2
    closest = min(segment_frames, key=lambda frame: abs(frame.timestamp - midpoint))
    return [closest.frame_id]


def _initial_modality_weight(transcript_text: str, visual_summary: str, representative_ids: list[str]) -> dict[str, float]:
    text_score = min(1.0, len(transcript_text.strip()) / 220) if transcript_text.strip() else 0.0
    vision_score = 0.15 if representative_ids else 0.0
    if visual_summary.strip():
        vision_score = max(vision_score, min(0.8, len(visual_summary.strip()) / 180))
    total = text_score + vision_score
    if total <= 0:
        return {"text": 0.0, "vision": 0.0}
    return {"text": round(text_score / total, 3), "vision": round(vision_score / total, 3)}


def _has_topic_marker(text: str) -> bool:
    lower = text.lower()
    return any(marker.lower() in lower for marker in TOPIC_MARKERS)
