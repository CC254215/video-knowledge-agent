from __future__ import annotations

from app.models import ModalityMode, ModalityProfile, MultimodalSegment, TranscriptSegment, VideoMetadata

VISUAL_HINTS = ("教程", "演示", "代码", "ppt", "白板", "图表", "操作", "实验", "安装", "配置", "demo", "slide")
QUESTION_VISUAL_HINTS = ("画面", "图中", "表格", "界面", "按钮", "代码", "公式", "演示", "操作界面", "点击", "视频里这一步", "截图", "ppt", "白板", "鐢婚潰", "鍥句腑", "琛ㄦ牸", "鐣岄潰", "鎸夐挳", "浠ｇ爜", "鍏紡", "婕旂ず")
NON_VISUAL_ACTION_HINTS = ("操作建议", "操作上的建议", "实操建议", "行动建议", "风控建议", "交易建议", "建议")


def route_modality(metadata: VideoMetadata, segments: list[TranscriptSegment]) -> ModalityProfile:
    transcript_text = " ".join(segment.text for segment in segments)
    word_count = len(transcript_text)
    spoken_seconds = sum(max(0.0, segment.end - segment.start) for segment in segments)
    duration = metadata.duration or spoken_seconds or 1.0
    speech_density = spoken_seconds / duration if duration else 0.0
    haystack = f"{metadata.title} {transcript_text}".lower()
    visual_score = 0.65 if any(hint.lower() in haystack for hint in VISUAL_HINTS) else 0.15

    if word_count >= 600 and speech_density >= 0.45 and visual_score < 0.5:
        mode = ModalityMode.text_dominant
        reason = "字幕/转录文本充足，主要信息可由 speech 证据支撑。"
    elif visual_score >= 0.5:
        mode = ModalityMode.vision_supportive
        reason = "标题或转录出现教程/演示/图表等视觉线索，建议按需使用关键帧和 frame_caption。"
    else:
        mode = ModalityMode.text_dominant
        reason = "未检测到强视觉依赖，当前阶段优先使用 speech。"

    return ModalityProfile(
        mode=mode,
        speech_density=round(speech_density, 3),
        visual_required_score=visual_score,
        ocr_required_score=0.0,
        reason=reason,
    )


def question_needs_visual_check(question: str) -> bool:
    lower = question.lower()
    if any(hint in question for hint in NON_VISUAL_ACTION_HINTS) and not any(
        visual in question for visual in ("画面", "图中", "界面", "按钮", "点击", "演示", "视频里这一步", "操作界面")
    ):
        return False
    return any(hint.lower() in lower for hint in QUESTION_VISUAL_HINTS)


def compute_segment_modality_weight(segment: MultimodalSegment, question: str | None = None) -> dict[str, float]:
    text = segment.transcript_text.strip()
    caption_text = segment.visual_summary.strip() or " ".join(caption.caption for caption in segment.visual_captions if caption.caption)
    text_score = min(1.0, len(text) / 220) if text else 0.0
    vision_score = 0.15 if segment.representative_frame_ids else 0.0
    if caption_text:
        vision_score = max(vision_score, min(0.8, len(caption_text) / 180))
    haystack = f"{text} {caption_text}".lower()
    if any(hint.lower() in haystack for hint in VISUAL_HINTS):
        vision_score = max(vision_score, 0.35)
    if question and question_needs_visual_check(question):
        vision_score = max(vision_score, 0.45)
    total = text_score + vision_score
    if total <= 0:
        return {"text": 0.0, "vision": 0.0}
    return {"text": round(text_score / total, 3), "vision": round(vision_score / total, 3)}

