from __future__ import annotations

from typing import Any

from app.models import ConversationTurn, Storyline, SummaryReport


def format_summary_for_ui(summary: SummaryReport | None) -> str:
    if summary is None:
        return "<div class='empty-card'>处理完成后将在这里显示结构化摘要。</div>"
    lines = ["<div class='section-title'>30 秒速览</div>"]
    lines.extend(f"<div class='bullet'>• {item}</div>" for item in summary.quick_overview)
    lines.append("<div class='section-title'>结构化大纲</div>")
    for item in summary.structured_outline:
        lines.append(f"<div class='outline-item'><span class='time-chip'>{item.timestamp}</span><b>{item.topic}</b></div>")
        lines.extend(f"<div class='sub-bullet'>- {point}</div>" for point in item.key_points[:3])
    if summary.warnings:
        lines.append("<div class='section-title warning'>Warnings</div>")
        lines.extend(f"<div class='sub-bullet'>- {warning}</div>" for warning in summary.warnings[:6])
    return "\n".join(lines)


def format_storyline_for_ui(storyline: Storyline | None) -> str:
    if storyline is None:
        return "<div class='empty-card'>处理完成后将生成 Query-Guided Storyline。</div>"
    if not storyline.nodes:
        return "<div class='empty-card'>暂无 storyline 节点。</div>"
    lines = ["<div class='timeline'>"]
    for node in storyline.nodes:
        timestamp = _seconds_to_ts(node.time_start)
        title = node.title or node.topic
        lines.append(
            "<div class='timeline-node'>"
            f"<div class='timeline-dot'></div><div><span class='time-chip'>{timestamp}</span> "
            f"<b>{title}</b><div class='muted'>{(node.summary or node.claim)[:120]}</div>"
            f"<div class='status-chip'>{node.status.value}</div></div></div>"
        )
    lines.append("</div>")
    return "\n".join(lines)


def format_suggested_questions_for_ui(summary: SummaryReport | None, turn: ConversationTurn | None = None) -> str:
    questions = turn.suggested_followup_questions if turn else (summary.open_questions if summary else [])
    if not questions:
        questions = ["这个视频的核心观点是什么？", "有哪些关键证据？", "有哪些可执行建议？"]
    return "\n".join(f"<button class='question-pill'>{question}</button>" for question in questions[:5])


def format_chat_answer_for_ui(answer: ConversationTurn | Any) -> str:
    if answer is None:
        return ""
    confidence = answer.confidence.value if hasattr(answer.confidence, "value") else answer.confidence
    return (
        f"{answer.answer}\n\n"
        f"**置信度：** `{confidence}`  \n"
        f"**是否需要视觉检查：** `{str(answer.needs_visual_check).lower()}`  \n"
        f"**Agreement：** `{getattr(answer, 'agreement_score', 0.0):.2f}`  \n"
        f"**原因：** {answer.reason}"
    )


def format_evidence_for_ui(answer: ConversationTurn | Any) -> str:
    if answer is None:
        return "<div class='empty-card'>回答后会在这里显示引用证据。</div>"
    evidence = getattr(answer, "evidence", []) or []
    if not evidence:
        return "<div class='empty-card'>本次回答没有可展示的视频证据。</div>"
    lines = ["<div class='evidence-list'>"]
    for item in evidence[:10]:
        lines.append(format_single_evidence_card(item))
    lines.append("</div>")
    lines.append("<div class='muted small'>OCR 当前禁用；证据仅来自 speech / frame / frame_caption。</div>")
    return "\n".join(lines)


def format_single_evidence_card(item: dict[str, Any]) -> str:
    evidence_type = item.get("evidence_type", "")
    timestamp = item.get("timestamp") or _seconds_to_ts(float(item.get("start", 0) or 0))
    start = item.get("time_start") or _seconds_to_ts(float(item.get("start", 0) or 0))
    end = item.get("time_end") or _seconds_to_ts(float(item.get("end", item.get("start", 0)) or 0))
    text = _escape(str(item.get("text", "")).strip())[:900]
    caption = _escape(str(item.get("frame_caption", "")).strip())[:700]
    frame_id = item.get("frame_id") or ""
    score = item.get("score")
    score_text = f"<span class='score-chip'>{float(score):.2f}</span>" if isinstance(score, (int, float)) else ""
    lines = [
        "<div class='evidence-card'>",
        f"<div class='evidence-head'><span class='time-chip'>{timestamp}</span><span class='type-chip'>{evidence_type}</span>{score_text}</div>",
        f"<div class='muted'>{start} - {end} · segment={item.get('segment_id', '')}</div>",
    ]
    if text:
        lines.append(f"<div class='evidence-text'><b>文本片段</b><br>{text}</div>")
    if caption:
        lines.append(f"<div class='evidence-text'><b>画面描述</b><br>{caption}</div>")
    if frame_id:
        lines.append(f"<div class='muted'>frame_id={frame_id}</div>")
    lines.append("</div>")
    return "\n".join(lines)


def format_status_for_ui(result: Any) -> str:
    if not result:
        return "<div class='status-card'>等待输入</div>"
    if getattr(result, "success", False):
        metadata = getattr(result, "metadata", None)
        title = getattr(metadata, "title", "") if metadata else ""
        author = getattr(metadata, "author", "") if metadata else ""
        duration = getattr(metadata, "duration", "") if metadata else ""
        modality = getattr(getattr(result, "modality_profile", None), "mode", "")
        return (
            "<div class='status-grid'>"
            f"<div><span>video_id</span><b>{getattr(result, 'video_id', '')}</b></div>"
            f"<div><span>标题</span><b>{_escape(title)[:80]}</b></div>"
            f"<div><span>作者</span><b>{_escape(author or '')}</b></div>"
            f"<div><span>时长</span><b>{duration or '--'}s</b></div>"
            f"<div><span>modality</span><b>{modality}</b></div>"
            f"<div><span>Obsidian</span><b>{_escape(getattr(result, 'obsidian_status', ''))}</b></div>"
            "</div>"
        )
    return f"<div class='error-card'>处理失败：{_escape(getattr(result, 'error', 'unknown error'))}</div>"


def format_evidence_gallery_items(evidence: list[dict[str, Any]]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for item in evidence or []:
        path = item.get("image_path")
        if not path:
            continue
        caption = item.get("frame_caption") or item.get("text") or item.get("timestamp") or "frame evidence"
        label = f"{item.get('timestamp', '')} · {item.get('evidence_type', 'frame')}\n{str(caption)[:120]}"
        if (path, label) not in items:
            items.append((path, label))
    return items[:12]


def _seconds_to_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
