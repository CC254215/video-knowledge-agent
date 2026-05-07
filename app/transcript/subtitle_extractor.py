from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def extract_subtitle_entries_from_ytdlp_info(info: dict[str, Any]) -> list[dict[str, object]]:
    """Return subtitle entries when yt-dlp exposes already downloaded JSON subtitles.

    First MVP keeps this conservative. Real subtitle file downloading/parsing can be expanded per
    platform without changing downstream transcript segment models.
    """
    subtitles = info.get("subtitles") or info.get("automatic_captions") or {}
    for tracks in subtitles.values():
        for track in tracks:
            if track.get("ext") == "json3" and track.get("filepath"):
                return parse_json3_subtitle(Path(track["filepath"]))
    return []


def parse_subtitle_file(path: str | Path) -> list[dict[str, object]]:
    subtitle_path = Path(path)
    suffix = subtitle_path.suffix.lower()
    if suffix == ".json3":
        return parse_json3_subtitle(subtitle_path)
    if suffix == ".vtt":
        return parse_vtt_subtitle(subtitle_path)
    if suffix == ".srt":
        return parse_srt_subtitle(subtitle_path)
    return []


def parse_json3_subtitle(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries: list[dict[str, object]] = []
    for event in data.get("events", []):
        if "segs" not in event:
            continue
        text = "".join(seg.get("utf8", "") for seg in event["segs"])
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        start = float(event.get("tStartMs", 0)) / 1000
        duration = float(event.get("dDurationMs", 0)) / 1000
        entries.append({"start": start, "end": start + duration, "text": text})
    return entries


def parse_vtt_subtitle(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    entries: list[dict[str, object]] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((idx for idx, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        start, end = _parse_timing_line(lines[timing_index])
        body = _clean_subtitle_text(" ".join(lines[timing_index + 1 :]))
        if body:
            entries.append({"start": start, "end": end, "text": body})
    return entries


def parse_srt_subtitle(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    entries: list[dict[str, object]] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((idx for idx, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        start, end = _parse_timing_line(lines[timing_index])
        body = _clean_subtitle_text(" ".join(lines[timing_index + 1 :]))
        if body:
            entries.append({"start": start, "end": end, "text": body})
    return entries


def _parse_timing_line(line: str) -> tuple[float, float]:
    left, right = line.split("-->", 1)
    return _parse_timestamp(left.strip()), _parse_timestamp(right.split()[0].strip())


def _parse_timestamp(value: str) -> float:
    value = value.replace(",", ".")
    parts = value.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = "0", parts[0], parts[1]
    else:
        return 0.0
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _clean_subtitle_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{\\.*?\}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
