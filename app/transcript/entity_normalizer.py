"""Lightweight, auditable normalization of noisy ASR named entities.

The LLM is only asked for entity mappings. Text replacement itself is deterministic so
ordinary speech is never paraphrased and raw ASR remains available as primary evidence.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from app.models import CanonicalEntity, MultimodalSegment, SummaryReport, VideoMetadata

_LATIN_TOKEN = re.compile(r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9]*(?:[ _-][A-Za-z][A-Za-z0-9]*)*)(?![A-Za-z0-9])")
_HIGH_VALUE_TYPES = {"software", "product", "model", "library", "framework", "company", "person", "version", "tool", "platform"}


def normalize_segments(
    segments: list[MultimodalSegment],
    metadata: VideoMetadata,
    summary: SummaryReport | None = None,
    output_path: str | Path | None = None,
    auto_apply_threshold: float = 0.85,
) -> tuple[list[MultimodalSegment], list[CanonicalEntity]]:
    """Apply high-confidence canonical entity mappings locally.

    Existing summary entities are preferred. For resumed/legacy summaries without the
    new field, a conservative title-vs-ASR similarity fallback supplies candidates.
    """
    entities = list(summary.canonical_entities if summary else [])
    entities = _merge_entities(entities, _infer_title_entities(metadata, segments))
    applied_rows: list[dict[str, object]] = []
    for segment in segments:
        raw = segment.raw_transcript_text or segment.transcript_text
        normalized = raw
        replacements: list[dict[str, object]] = []
        for entity in entities:
            if entity.needs_review or entity.confidence < auto_apply_threshold:
                continue
            for alias in sorted(set(entity.aliases), key=len, reverse=True):
                if not alias.strip() or alias.casefold() == entity.canonical.casefold():
                    continue
                replaced, count = _replace_token(normalized, alias, entity.canonical)
                if count:
                    normalized = replaced
                    replacements.append({"raw": alias, "canonical": entity.canonical, "confidence": entity.confidence})
        segment.raw_transcript_text = raw
        segment.normalized_transcript_text = normalized
        segment.normalization_applied = normalized != raw
        segment.replacements = replacements
        # Keep the existing field as the retrieval text for compatibility.
        segment.transcript_text = normalized
        if replacements:
            applied_rows.append({"segment_id": segment.segment_id, "replacements": replacements})
    if output_path:
        Path(output_path).write_text(
            json.dumps({"entities": [item.model_dump(mode="json") for item in entities], "applied": applied_rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return segments, entities


def _replace_token(text: str, alias: str, canonical: str) -> tuple[str, int]:
    # Latin aliases use word boundaries; CJK/space-containing aliases use explicit
    # case-insensitive matching while avoiding repeated replacement.
    flags = re.IGNORECASE
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]*", alias):
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])", flags)
    else:
        pattern = re.compile(re.escape(alias), flags)
    return pattern.subn(canonical, text)


def _merge_entities(primary: Iterable[CanonicalEntity], inferred: Iterable[CanonicalEntity]) -> list[CanonicalEntity]:
    merged: dict[str, CanonicalEntity] = {}
    for entity in [*primary, *inferred]:
        key = entity.canonical.casefold().strip()
        if not key:
            continue
        old = merged.get(key)
        if old is None or entity.confidence > old.confidence:
            merged[key] = entity
        elif old is not None:
            old.aliases = sorted(set(old.aliases + entity.aliases), key=str.casefold)
            old.evidence_sources = sorted(set(old.evidence_sources + entity.evidence_sources))
    return list(merged.values())


def _infer_title_entities(metadata: VideoMetadata, segments: list[MultimodalSegment]) -> list[CanonicalEntity]:
    title_tokens = [token.strip() for token in _LATIN_TOKEN.findall(metadata.title or "")]
    candidates: list[CanonicalEntity] = []
    asr_tokens: set[str] = set()
    for segment in segments:
        for token in _LATIN_TOKEN.findall(segment.raw_transcript_text or segment.transcript_text):
            if len(token.replace(" ", "")) >= 4:
                asr_tokens.add(token.strip())
    for canonical in title_tokens:
        compact = canonical.replace(" ", "")
        if len(compact) < 4 or canonical.isupper():
            continue
        aliases: list[str] = []
        for token in asr_tokens:
            ratio = SequenceMatcher(None, compact.casefold(), token.replace(" ", "").casefold()).ratio()
            # ASR often changes one vowel/consonant in a product name (e.g. a
            # phonetic "WalkBody" for a title's "Workbuddy"); the title plus
            # repeated occurrence is the evidence combination, not the edit
            # distance alone, that makes this candidate safe.
            if token.casefold() != canonical.casefold() and ratio >= 0.55:
                aliases.append(token)
        if aliases:
            candidates.append(CanonicalEntity(
                canonical=canonical,
                aliases=sorted(set(aliases), key=str.casefold),
                entity_type="named_entity",
                confidence=0.88,
                evidence_sources=["video_title", "speech"],
                reason="title entity is consistently similar to repeated ASR variants",
            ))
    return candidates
