from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path

from app.models import ConversationTurn, MultimodalSegment, QAAnswer, Storyline, SummaryReport, VideoMetadata
from app.reasoning.summarizer import format_timestamp

ILLEGAL_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1F]'


def safe_filename(value: str, max_len: int = 120) -> str:
    cleaned = re.sub(ILLEGAL_FILENAME_CHARS, "-", value).strip(" .-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return (cleaned or "untitled")[:max_len]


def _unique_path(path: Path, video_id: str) -> Path:
    if not path.exists():
        return path
    stamp = datetime.utcnow().strftime("%H%M%S")
    return path.with_name(f"{path.stem} - {video_id[:8]}-{stamp}{path.suffix}")


def ensure_vault_dirs(vault_path: Path) -> None:
    if not vault_path.exists():
        raise FileNotFoundError(f"OBSIDIAN_VAULT_PATH does not exist: {vault_path}")
    for subdir in (
        "00_Inbox/Video",
        "10_Sources/Videos",
        "20_Atomic_Notes",
        "30_Storylines",
        "40_MOCs",
        "90_Assets",
    ):
        (vault_path / subdir).mkdir(parents=True, exist_ok=True)


def _yaml_list(values: list[str]) -> str:
    return "[" + ", ".join(f'"{value}"' for value in values) + "]"


def render_video_note(
    metadata: VideoMetadata,
    summary: SummaryReport,
    storyline: Storyline,
    qa_history: list[QAAnswer] | None = None,
    multimodal_segments: list[MultimodalSegment] | None = None,
    conversation_history: list[ConversationTurn] | None = None,
) -> str:
    topics = [item.topic for item in summary.structured_outline[:8]]
    lines = [
        "---",
        "type: video-note",
        f"source: {metadata.source}",
        f"url: {metadata.url or ''}",
        f"author: {metadata.author or ''}",
        f"duration: {metadata.duration or ''}",
        f"processed_at: {datetime.utcnow().isoformat()}",
        f"modality: {summary.modality_note}",
        f"topics: {_yaml_list(topics)}",
        "confidence: medium",
        f"video_id: {metadata.video_id}",
        "---",
        "",
        "# 30 秒速览",
        "",
    ]
    lines.extend(f"- {item}" for item in summary.quick_overview)
    lines.extend(["", "# 结构化大纲", ""])
    for item in summary.structured_outline:
        lines.append(f"- {item.timestamp} {item.topic} ({', '.join(item.segment_ids)})")
        lines.extend(f"  - {point}" for point in item.key_points)
    lines.extend(["", "# 深度分析", ""])
    for item in summary.deep_analysis:
        lines.append(f"- {item.claim} [{', '.join(item.evidence_segment_ids)}]")
        lines.extend(f"  - Caveat: {caveat}" for caveat in item.caveats)
    lines.extend(["", "# 行动清单", ""])
    lines.extend(f"- {item.action} [{', '.join(item.evidence_segment_ids)}]" for item in summary.action_items)
    lines.extend(["", "# Query-Guided Storyline", ""])
    for node in storyline.nodes:
        lines.append(
            f"- {format_timestamp(node.time_start)} {node.topic}: {node.claim} "
            f"[speech={','.join(node.speech_evidence_ids)} frame={','.join(node.frame_evidence_ids)}] "
            f"status={node.status.value} uncertainty={node.uncertainty}"
        )
    lines.extend(["", "# 关键证据", ""])
    lines.extend(f"- {quote.timestamp} `{quote.segment_id}` {quote.quote}" for quote in summary.important_quotes)
    lines.extend(["", "# 多模态证据", ""])
    for segment in multimodal_segments or []:
        lines.append(
            f"- {format_timestamp(segment.start)} `{segment.segment_id}` types={','.join(segment.evidence_types)} "
            f"frames={','.join(segment.representative_frame_ids)}"
        )
    lines.extend(["", "# 关键视觉帧", ""])
    for segment in multimodal_segments or []:
        for frame_id in segment.representative_frame_ids:
            caption = next((item.caption for item in segment.visual_captions if item.frame_id == frame_id), "")
            lines.append(f"- {format_timestamp(segment.start)} frame_id={frame_id}, caption={caption}")
    lines.extend(["", "# OCR 摘录", "", "本阶段 OCR 已禁用。"])
    lines.extend(["", "# 用户对话记录", ""])
    if conversation_history:
        for turn in conversation_history:
            lines.append(f"- Q: {turn.user_question}")
            lines.append(f"  A: {turn.answer} [{', '.join(turn.evidence_ids)}] confidence={turn.confidence.value}")
    elif qa_history:
        for answer in qa_history:
            lines.append(f"- Q: {answer.question}")
            lines.append(f"  A: {answer.answer} [{', '.join(answer.evidence_segment_ids)}]")
    lines.extend(["", "# 建议追问问题", ""])
    lines.extend(f"- {question}" for question in summary.open_questions)
    lines.extend(["", "# 原始证据索引", ""])
    for quote in summary.important_quotes:
        lines.append(f"- {quote.timestamp} {quote.segment_id}: {quote.quote}")
    return "\n".join(lines).rstrip() + "\n"


def render_storyline_note(metadata: VideoMetadata, storyline: Storyline) -> str:
    lines = [f"# {metadata.title} - Storyline", "", f"Query: {storyline.query}", ""]
    for node in storyline.nodes:
        lines.append(f"## {format_timestamp(node.time_start)} {node.topic}")
        lines.append("")
        lines.append(node.claim)
        lines.append("")
        lines.append(f"Speech Evidence: {', '.join(node.speech_evidence_ids) or 'none'}")
        lines.append(f"Frame Evidence: {', '.join(node.frame_evidence_ids) or 'none'}")
        lines.append("OCR Evidence: disabled in current stage")
        lines.append(f"Status: {node.status.value}")
        lines.append(f"Uncertainty: {node.uncertainty}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_obsidian_notes(
    vault_path: Path,
    metadata: VideoMetadata,
    summary: SummaryReport,
    storyline: Storyline,
    qa_history: list[QAAnswer] | None = None,
    multimodal_segments: list[MultimodalSegment] | None = None,
    conversation_history: list[ConversationTurn] | None = None,
) -> dict[str, Path | str]:
    ensure_vault_dirs(vault_path)
    audit_trace_id = f"audit-{uuid.uuid4().hex[:12]}"
    video_folder = _video_export_dir(vault_path, metadata)
    video_folder.mkdir(parents=True, exist_ok=True)

    video_note = video_folder / "index.md"
    storyline_note = video_folder / "storyline.md"
    evidence_note = video_folder / "evidence.md"
    qa_note = video_folder / "qa.md"
    updates_note = video_folder / "updates.md"
    legacy_storyline_link = vault_path / "30_Storylines" / f"{metadata.video_id} - {safe_filename(metadata.title)}.md"

    _atomic_write(video_note, render_video_note(metadata, summary, storyline, qa_history, multimodal_segments, conversation_history))
    _atomic_write(storyline_note, render_storyline_note(metadata, storyline))
    _atomic_write(evidence_note, render_evidence_note(metadata, multimodal_segments or []))
    _atomic_write(qa_note, render_qa_note(metadata, qa_history, conversation_history))
    _append_update_block(updates_note, audit_trace_id, multimodal_segments or [], conversation_history or [])
    _atomic_write(legacy_storyline_link, _storyline_pointer(metadata, storyline_note))
    _update_video_index(vault_path, metadata, video_note)

    return {
        "video_note": video_note,
        "storyline_note": storyline_note,
        "evidence_note": evidence_note,
        "qa_note": qa_note,
        "updates_note": updates_note,
        "video_dir": video_folder,
        "audit_trace_id": audit_trace_id,
    }


def _video_export_dir(vault_path: Path, metadata: VideoMetadata) -> Path:
    title = safe_filename(metadata.title, max_len=80)
    return vault_path / "10_Sources" / "Videos" / f"{metadata.video_id} - {title}"


def render_evidence_note(metadata: VideoMetadata, multimodal_segments: list[MultimodalSegment]) -> str:
    lines = [
        "---",
        "type: video-evidence",
        f"video_id: {metadata.video_id}",
        f"source: {metadata.source}",
        f"url: {metadata.url or ''}",
        f"updated_at: {datetime.utcnow().isoformat()}",
        "---",
        "",
        f"# {metadata.title} - Evidence",
        "",
        "## Segments",
        "",
    ]
    for segment in multimodal_segments:
        lines.append(f"### {format_timestamp(segment.start)} `{segment.segment_id}`")
        lines.append("")
        lines.append(f"- time_range: {format_timestamp(segment.start)} - {format_timestamp(segment.end)}")
        lines.append(f"- evidence_types: {', '.join(segment.evidence_types)}")
        if segment.transcript_text.strip():
            lines.extend(["", "Speech:", "", segment.transcript_text.strip(), ""])
        if segment.representative_frame_ids:
            lines.append(f"- representative_frames: {', '.join(segment.representative_frame_ids)}")
        for caption in segment.visual_captions:
            if caption.caption.strip():
                lines.append(f"- frame_caption {format_timestamp(caption.timestamp)} `{caption.frame_id}`: {caption.caption}")
                if caption.image_path:
                    lines.append(f"  - image: {caption.image_path}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_qa_note(
    metadata: VideoMetadata,
    qa_history: list[QAAnswer] | None = None,
    conversation_history: list[ConversationTurn] | None = None,
) -> str:
    lines = [
        "---",
        "type: video-qa",
        f"video_id: {metadata.video_id}",
        f"source: {metadata.source}",
        f"url: {metadata.url or ''}",
        f"updated_at: {datetime.utcnow().isoformat()}",
        "---",
        "",
        f"# {metadata.title} - Q&A",
        "",
    ]
    if conversation_history:
        for turn in conversation_history:
            lines.append(f"## Q: {turn.user_question}")
            lines.append("")
            lines.append(turn.answer)
            lines.append("")
            lines.append(f"- evidence: {', '.join(turn.evidence_ids)}")
            lines.append(f"- confidence: {turn.confidence.value}")
            lines.append(f"- created_at: {turn.created_at.isoformat()}")
            lines.append("")
    elif qa_history:
        for answer in qa_history:
            lines.append(f"## Q: {answer.question}")
            lines.append("")
            lines.append(answer.answer)
            lines.append("")
            lines.append(f"- evidence: {', '.join(answer.evidence_segment_ids)}")
            lines.append(f"- confidence: {answer.confidence.value}")
            lines.append("")
    else:
        lines.append("_No Q&A exported yet._")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _storyline_pointer(metadata: VideoMetadata, storyline_note: Path) -> str:
    return "\n".join(
        [
            "---",
            "type: video-storyline-pointer",
            f"video_id: {metadata.video_id}",
            f"updated_at: {datetime.utcnow().isoformat()}",
            "---",
            "",
            f"# {metadata.title} - Storyline",
            "",
            f"Canonical note: [[{storyline_note.parent.name}/storyline|storyline]]",
            "",
        ]
    )


def _update_video_index(vault_path: Path, metadata: VideoMetadata, video_note: Path) -> None:
    index_path = vault_path / "40_MOCs" / "Video Index.md"
    entry = f"- [[{video_note.parent.name}/index|{metadata.title}]] `video_id={metadata.video_id}` source={metadata.source} duration={metadata.duration or ''}"
    existing_lines: list[str] = []
    if index_path.exists():
        existing_lines = index_path.read_text(encoding="utf-8").splitlines()
    else:
        existing_lines = ["# Video Index", ""]
    filtered = [line for line in existing_lines if f"video_id={metadata.video_id}`" not in line]
    if filtered and filtered[-1].strip():
        filtered.append("")
    filtered.append(entry)
    _atomic_write(index_path, "\n".join(filtered).rstrip() + "\n")


def _find_existing_video_note(vault_path: Path, video_id: str) -> Path | None:
    for path in (vault_path / "10_Sources" / "Videos").glob("*/index.md"):
        try:
            if f"video_id: {video_id}" in path.read_text(encoding="utf-8"):
                return path
        except OSError:
            continue
    for path in (vault_path / "10_Sources" / "Videos").glob("*.md"):
        try:
            if f"video_id: {video_id}" in path.read_text(encoding="utf-8"):
                return path
        except OSError:
            continue
    return None


def _append_update_block(
    updates_note: Path,
    audit_trace_id: str,
    multimodal_segments: list[MultimodalSegment],
    conversation_history: list[ConversationTurn],
) -> None:
    current = updates_note.read_text(encoding="utf-8") if updates_note.exists() else _updates_header()
    if audit_trace_id in current:
        return
    _atomic_write(updates_note, current.rstrip() + "\n\n" + _incremental_block(audit_trace_id, multimodal_segments, conversation_history))


def _updates_header() -> str:
    return "---\ntype: video-updates\n---\n\n# Updates\n"


def _append_incremental_block(
    video_note: Path,
    audit_trace_id: str,
    multimodal_segments: list[MultimodalSegment],
    conversation_history: list[ConversationTurn],
) -> None:
    current = video_note.read_text(encoding="utf-8")
    if audit_trace_id in current:
        return
    _atomic_write(video_note, current.rstrip() + "\n\n" + _incremental_block(audit_trace_id, multimodal_segments, conversation_history))


def _incremental_block(
    audit_trace_id: str,
    multimodal_segments: list[MultimodalSegment],
    conversation_history: list[ConversationTurn],
) -> str:
    lines = [
        f"<!-- audit_trace_id: {audit_trace_id} -->",
        f"## Incremental Evidence Update - {audit_trace_id}",
        "",
        f"- updated_at: {datetime.utcnow().isoformat()}",
        "- rollback: append-only block; previous note content preserved",
        "",
        "### New Evidence",
    ]
    seen: set[str] = set()
    for segment in multimodal_segments:
        for caption in segment.visual_captions:
            key = f"{segment.segment_id}:{caption.frame_id}:{caption.timestamp}"
            if key in seen or not caption.caption.strip():
                continue
            seen.add(key)
            lines.append(
                f"- {format_timestamp(caption.timestamp)} `{segment.segment_id}` frame_id={caption.frame_id} "
                f"caption={caption.caption} image={caption.image_path or ''}"
            )
    lines.extend(["", "### User Annotations"])
    for turn in conversation_history:
        lines.append(f"- Q: {turn.user_question}")
        lines.append(f"  A: {turn.answer} evidence={','.join(turn.evidence_ids)} confidence={turn.confidence.value}")
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)
