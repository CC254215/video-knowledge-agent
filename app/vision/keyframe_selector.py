from __future__ import annotations

import random

from app.models import TranscriptSegment, VideoFrame


def select_representative_frames(
    transcript_segments: list[TranscriptSegment],
    candidate_frames: list[VideoFrame],
    max_per_segment: int = 1,
    min_representative_frames: int = 8,
    min_global_gap_seconds: float = 9.0,
    strategy: str = "default",
) -> dict[str, list[str]]:
    """Select representative frame ids for each transcript segment.

    OCR is disabled in the current stage, so selection is based only on timestamp coverage and
    lightweight visual diversity already handled by frame dedupe.
    """
    selected_by_segment: dict[str, list[str]] = {}
    selected_timestamps: list[float] = []
    for segment in transcript_segments:
        frames = [frame for frame in candidate_frames if segment.start <= frame.timestamp <= segment.end]
        if not frames:
            selected_by_segment[segment.segment_id] = []
            continue
        if strategy == "random":
            rng = random.Random(f"{segment.segment_id}:{segment.start}:{segment.end}")
            ranked = list(frames)
            rng.shuffle(ranked)
        else:
            midpoint = (segment.start + segment.end) / 2
            ranked = sorted(frames, key=lambda frame: abs(frame.timestamp - midpoint))
        chosen = _pick_with_global_gap(ranked, selected_timestamps, max_count=max_per_segment, min_gap=min_global_gap_seconds)
        for frame in chosen:
            frame.selected = True
            frame.selection_reason = "representative_for_transcript_segment"
            selected_timestamps.append(frame.timestamp)
        selected_by_segment[segment.segment_id] = [frame.frame_id for frame in chosen]
    _ensure_minimum_representatives(
        selected_by_segment,
        transcript_segments,
        candidate_frames,
        min_representative_frames,
        min_global_gap_seconds,
    )
    return selected_by_segment


def _ensure_minimum_representatives(
    selected_by_segment: dict[str, list[str]],
    transcript_segments: list[TranscriptSegment],
    candidate_frames: list[VideoFrame],
    min_representative_frames: int,
    min_global_gap_seconds: float,
) -> None:
    if len(candidate_frames) < min_representative_frames:
        return
    selected_ids = {frame_id for ids in selected_by_segment.values() for frame_id in ids}
    if len(selected_ids) >= min_representative_frames:
        return
    needed = min_representative_frames - len(selected_ids)
    selected_timestamps = [
        frame.timestamp for frame in candidate_frames if frame.frame_id in selected_ids
    ]
    candidates = [frame for frame in candidate_frames if frame.frame_id not in selected_ids]
    if not candidates:
        return
    step = max(1, len(candidates) // max(1, needed))
    spaced_candidates = candidates[::step]
    additions = _pick_with_global_gap(
        spaced_candidates,
        selected_timestamps,
        max_count=needed,
        min_gap=min_global_gap_seconds,
    )
    for frame in additions:
        owner = next((segment.segment_id for segment in transcript_segments if segment.start <= frame.timestamp <= segment.end), None)
        if owner is None and transcript_segments:
            owner = min(transcript_segments, key=lambda segment: abs(((segment.start + segment.end) / 2) - frame.timestamp)).segment_id
        if owner is None:
            continue
        if selected_by_segment.get(owner):
            continue
        frame.selected = True
        frame.selection_reason = "minimum_representative_frame_coverage"
        selected_timestamps.append(frame.timestamp)
        selected_by_segment[owner] = [frame.frame_id]


def _pick_with_global_gap(
    candidates: list[VideoFrame],
    existing_timestamps: list[float],
    max_count: int,
    min_gap: float,
) -> list[VideoFrame]:
    chosen: list[VideoFrame] = []
    all_timestamps = list(existing_timestamps)
    for frame in candidates:
        if len(chosen) >= max_count:
            break
        if any(abs(frame.timestamp - timestamp) < min_gap for timestamp in all_timestamps):
            continue
        chosen.append(frame)
        all_timestamps.append(frame.timestamp)
    return chosen
