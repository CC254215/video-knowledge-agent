from __future__ import annotations

import re
from collections.abc import Iterable

from app.models import TranscriptSegment
from app.transcript.normalizer import normalize_asr_text

RawTranscriptItem = dict[str, object] | TranscriptSegment

TOPIC_MARKERS = (
    "接下来",
    "第二点",
    "第三点",
    "总结一下",
    "但是",
    "另一个问题",
    "another point",
    "next",
    "in summary",
    "however",
)

MAX_SEGMENT_CHARS = 1800


def _as_segmentish(item: RawTranscriptItem) -> tuple[float, float, str, str | None, float | None]:
    if isinstance(item, TranscriptSegment):
        return item.start, item.end, item.text, item.speaker, item.asr_confidence
    return (
        float(item.get("start", 0.0)),
        float(item.get("end", item.get("start", 0.0))),
        normalize_asr_text(str(item.get("text", ""))),
        item.get("speaker") if isinstance(item.get("speaker"), str) else None,
        float(item["asr_confidence"]) if item.get("asr_confidence") is not None else None,
    )


def extract_keywords(text: str, limit: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text)
    stopwords = {"这个", "我们", "一个", "以及", "然后", "所以", "the", "and", "that", "with", "for"}
    counts: dict[str, int] = {}
    for token in tokens:
        norm = token.lower()
        if norm in stopwords:
            continue
        counts[norm] = counts.get(norm, 0) + 1
    return [word for word, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]


def extract_entities(text: str, limit: int = 8) -> list[str]:
    entities = re.findall(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,3}\b", text)
    seen: list[str] = []
    for entity in entities:
        if entity not in seen:
            seen.append(entity)
    return seen[:limit]


def _has_topic_marker(text: str) -> bool:
    lower = text.lower()
    return any(marker.lower() in lower for marker in TOPIC_MARKERS)


def _make_segment(index: int, parts: list[tuple[float, float, str, str | None, float | None]]) -> TranscriptSegment:
    start = parts[0][0]
    end = max(part[1] for part in parts)
    text = " ".join(part[2].strip() for part in parts if part[2].strip())
    speaker = parts[0][3]
    confidences = [part[4] for part in parts if part[4] is not None]
    confidence = sum(confidences) / len(confidences) if confidences else None
    return TranscriptSegment(
        segment_id=f"seg_{index:04d}",
        start=start,
        end=end,
        speaker=speaker,
        text=text[:MAX_SEGMENT_CHARS],
        asr_confidence=confidence,
        keywords=extract_keywords(text),
        entities=extract_entities(text),
    )


def segment_transcript(
    transcript_items: Iterable[RawTranscriptItem],
    min_seconds: float = 60.0,
    max_seconds: float = 120.0,
) -> list[TranscriptSegment]:
    normalized = [_as_segmentish(item) for item in transcript_items if _as_segmentish(item)[2].strip()]
    if not normalized:
        return []

    segments: list[TranscriptSegment] = []
    bucket: list[tuple[float, float, str, str | None, float | None]] = []

    for item in normalized:
        if not bucket:
            bucket.append(item)
            continue

        start = bucket[0][0]
        proposed_duration = item[1] - start
        proposed_chars = len(" ".join(part[2] for part in bucket)) + len(item[2])
        marker_split = proposed_duration >= min_seconds and _has_topic_marker(item[2])
        hard_split = proposed_duration >= max_seconds or proposed_chars > MAX_SEGMENT_CHARS

        if marker_split or hard_split:
            segments.append(_make_segment(len(segments), bucket))
            bucket = [item]
        else:
            bucket.append(item)

    if bucket:
        segments.append(_make_segment(len(segments), bucket))
    return segments
