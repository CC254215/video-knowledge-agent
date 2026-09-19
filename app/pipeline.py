from __future__ import annotations

import json
import hashlib
import inspect
import logging
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from app.config import Settings, get_settings
from app.ingestion.downloader import DownloadError, download_audio_for_asr, download_subtitle_file, download_video_file, fetch_url_metadata
from app.ingestion.local_file import metadata_from_local_file
from app.ingestion.url_safety import validate_public_media_url
from app.memory.mempalace_adapter import create_memory_adapter
from app.memory.obsidian_writer import write_obsidian_notes
from app.modality.router import route_modality
from app.models import (
    FrameCaption,
    ModalityProfile,
    MultimodalSegment,
    PartialPipelineRestartResult,
    RefineRange,
    Storyline,
    SummaryReport,
    TranscriptSegment,
    VideoFrame,
    VideoCorrelationProfile,
    VideoVisualProfile,
    VideoMetadata,
    VideoTranscript,
)
from app.reasoning.conversation_agent import ConversationAgent
from app.reasoning.qa import answer_question
from app.reasoning.storyline import build_storyline_from_multimodal_segments, build_storyline_from_segments, load_storyline, save_storyline
from app.reasoning.summarizer import summarize_multimodal_video, summarize_video
from app.retrieval.chroma_memory import ChromaMemoryStore
from app.retrieval.vector_store import InMemoryVectorStore
from app.runtime.pipeline_runtime import apply_dynamic_llm_timeout, run_stage_with_restarts
from app.runtime.processing_state import ProcessingState
from app.storage.sqlite_store import SQLiteStore
from app.transcript.asr import ASRUnavailableError, get_asr_adapter
from app.transcript.multimodal_segmenter import build_multimodal_segments, load_multimodal_segments, save_multimodal_segments
from app.transcript.entity_normalizer import normalize_segments
from app.transcript.normalizer import normalize_asr_text
from app.transcript.segmenter import segment_transcript
from app.transcript.subtitle_extractor import parse_subtitle_file
from app.vision.frame_refiner import FrameEvidenceRefiner
from app.vision.frame_deduper import dedupe_frames
from app.vision.frame_extractor import extract_frames
from app.vision.keyframe_selector import select_representative_frames
from app.vision.correlation_agent import assess_video_text_correlation
from app.vision.video_classifier import classify_video_visual_intensity, save_visual_profile
from app.vision.vlm_captioner import VLMCaptioner, save_frame_captions

logger = logging.getLogger(__name__)
_RUN_REBUILT_MULTIMODAL: set[str] = set()
_RUN_MULTIMODAL_CACHE: dict[str, list[MultimodalSegment]] = {}
_CACHE_LOCK = threading.RLock()
ProgressCallback = Callable[[str], None]


def _progress(callback: ProgressCallback | None, message: str) -> None:
    if callback:
        callback(message)


def video_dir(video_id: str, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    path = settings.videos_dir / video_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, model_or_payload: object) -> None:
    if hasattr(model_or_payload, "model_dump_json"):
        path.write_text(model_or_payload.model_dump_json(indent=2), encoding="utf-8")  # type: ignore[attr-defined]
    else:
        path.write_text(json.dumps(model_or_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_metadata(video_id: str, settings: Settings | None = None) -> VideoMetadata:
    return VideoMetadata.model_validate(json.loads((video_dir(video_id, settings) / "metadata.json").read_text(encoding="utf-8")))


def read_segments(video_id: str, settings: Settings | None = None) -> list[TranscriptSegment]:
    transcript = VideoTranscript.model_validate(json.loads((video_dir(video_id, settings) / "transcript.json").read_text(encoding="utf-8")))
    for segment in transcript.segments:
        segment.text = normalize_asr_text(segment.text)
        segment.keywords = [normalize_asr_text(keyword) for keyword in segment.keywords]
    return transcript.segments


def read_transcript_source(video_id: str, settings: Settings | None = None) -> str:
    transcript = VideoTranscript.model_validate(json.loads((video_dir(video_id, settings) / "transcript.json").read_text(encoding="utf-8")))
    return transcript.source


def read_multimodal_segments(video_id: str, settings: Settings | None = None) -> list[MultimodalSegment]:
    path = video_dir(video_id, settings) / "multimodal_segments.json"
    return load_multimodal_segments(path) if path.exists() else []


def _cache_key(video_id: str, settings: Settings) -> str:
    return f"{settings.data_dir.resolve()}::{video_id}"


def create_mock_transcript() -> list[dict[str, object]]:
    texts = [
        "今天我们讨论 Video Knowledge Agent。它不是普通摘要工具，而是主动摄取长视频信息并生成可追溯知识笔记。",
        "第一点，系统必须优先获取字幕；没有字幕时才使用 ASR，这样可以降低成本并提高稳定性。",
        "接下来，分段需要保留时间戳证据。每一个关键结论都要绑定 segment_id，方便之后问答和 Obsidian 回链。",
        "但是，视觉理解不应该一开始就做重。访谈、播客、讲座和评论类内容通常文本足够，只有教程、代码、PPT 或界面操作才需要关键帧和视觉描述。",
        "另一个问题是回答可信度。我们可以对用户 query 和候选答案分别选择证据，再计算交集比例作为一致性信号。",
        "总结一下，第一版应该跑通 mock transcript 到 summary、storyline、Obsidian 导出的最小闭环，然后再接真实视频 URL、ASR 和数据库。",
    ]
    return [{"start": i * 55.0, "end": i * 55.0 + 50.0, "text": text} for i, text in enumerate(texts)]


def ingest_mock(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    metadata = VideoMetadata(
        video_id="mock_video_001",
        title="Video Knowledge Agent MVP Mock Lecture",
        author="Local Mock",
        source="mock",
        duration=360,
        language="zh",
    )
    segments = segment_transcript(create_mock_transcript())
    transcript = VideoTranscript(video_id=metadata.video_id, segments=segments, source="mock", language="zh")
    out_dir = video_dir(metadata.video_id, settings)
    write_json(out_dir / "metadata.json", metadata)
    write_json(out_dir / "transcript.json", transcript)
    ensure_multimodal_segments(metadata.video_id, settings)
    return metadata.video_id


def ingest_local_file(
    file_path: Path,
    settings: Settings | None = None,
    progress_callback: ProgressCallback | None = None,
    timing_collector: dict[str, float] | None = None,
) -> str:
    settings = settings or get_settings()
    metadata = metadata_from_local_file(file_path)
    out_dir = video_dir(metadata.video_id, settings)
    metadata_path = out_dir / "metadata.json"
    transcript_path = out_dir / "transcript.json"
    if metadata_path.exists() and transcript_path.exists() and not settings.force_refresh:
        logger.info("Reusing existing transcript for video_id=%s", metadata.video_id)
        _normalize_existing_transcript(transcript_path)
        return metadata.video_id
    try:
        logger.info("Running ASR for local file: %s", file_path)
        _progress(progress_callback, "正在进行语音识别 0.0%")
        asr_started = time.monotonic()
        adapter = get_asr_adapter(settings)
        raw_items = (
            adapter.transcribe(file_path, progress_callback=progress_callback)
            if progress_callback
            else adapter.transcribe(file_path)
        )
        _record_timing(timing_collector, "asr", asr_started)
    except ASRUnavailableError as exc:
        raise RuntimeError(f"No subtitles for local file and ASR is unavailable: {exc}") from exc
    segments = segment_transcript(raw_items)
    write_json(out_dir / "metadata.json", metadata)
    write_json(out_dir / "transcript.json", VideoTranscript(video_id=metadata.video_id, segments=segments, source="asr"))
    _progress(progress_callback, "正在提取关键帧和构建多模态证据")
    multimodal_started = time.monotonic()
    ensure_multimodal_segments(metadata.video_id, settings)
    _record_timing(timing_collector, "multimodal_preprocess", multimodal_started)
    return metadata.video_id


def _normalize_existing_transcript(transcript_path: Path) -> None:
    transcript = VideoTranscript.model_validate(json.loads(transcript_path.read_text(encoding="utf-8")))
    changed = False
    for segment in transcript.segments:
        normalized = normalize_asr_text(segment.text)
        if normalized != segment.text:
            segment.text = normalized
            segment.keywords = [normalize_asr_text(keyword) for keyword in segment.keywords]
            changed = True
    if changed:
        write_json(transcript_path, transcript)


def ingest_url(
    url: str,
    settings: Settings | None = None,
    progress_callback: ProgressCallback | None = None,
    timing_collector: dict[str, float] | None = None,
) -> str:
    settings = settings or get_settings()
    url = url.strip()
    cached_video_id = _cached_video_id_from_url(url, settings)
    if cached_video_id and not settings.force_refresh:
        logger.info("Reusing cached URL ingest artifacts for video_id=%s", cached_video_id)
        return cached_video_id
    url = validate_public_media_url(url)
    _preflight_url_runtime(settings)
    _progress(progress_callback, "正在读取视频信息")
    metadata_started = time.monotonic()
    try:
        metadata, _info = fetch_url_metadata(url, settings.data_dir / "downloads")
    except DownloadError as exc:
        raise RuntimeError(str(exc)) from exc
    _record_timing(timing_collector, "url_metadata", metadata_started)
    if metadata.duration and metadata.duration > settings.max_media_duration_seconds:
        raise RuntimeError(
            f"media_too_long: duration {metadata.duration:.0f}s exceeds limit {settings.max_media_duration_seconds:.0f}s"
        )
    out_dir = video_dir(metadata.video_id, settings)
    download_dir = out_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.json"
    transcript_path = out_dir / "transcript.json"
    if metadata_path.exists() and transcript_path.exists() and not settings.force_refresh:
        logger.info("Reusing cached URL ingest artifacts for video_id=%s", metadata.video_id)
        return metadata.video_id
    write_json(out_dir / "metadata.json", metadata)
    try:
        _progress(progress_callback, "正在下载视频 0.0%")
        video_download_started = time.monotonic()
        video_path = (
            download_video_file(url, download_dir, progress_callback=progress_callback)
            if progress_callback
            else download_video_file(url, download_dir)
        )
        _record_timing(timing_collector, "video_download", video_download_started)
        _validate_media_size(video_path, settings)
        metadata.local_path = str(video_path)
        write_json(out_dir / "metadata.json", metadata)
    except DownloadError as exc:
        if settings.strict_runtime:
            raise RuntimeError(str(exc)) from exc
        logger.warning("Video download failed; degraded fallback may lack trustworthy visual evidence: %s", exc)

    raw_items: list[dict[str, object]] = []
    source = "unknown"
    _progress(progress_callback, "正在获取字幕")
    subtitle_started = time.monotonic()
    subtitle_path = (
        download_subtitle_file(url, download_dir, progress_callback=progress_callback)
        if progress_callback
        else download_subtitle_file(url, download_dir)
    )
    _record_timing(timing_collector, "subtitle", subtitle_started)
    if subtitle_path:
        raw_items = parse_subtitle_file(subtitle_path)
        source = "subtitle"
    if not raw_items:
        try:
            _progress(progress_callback, "未找到可用字幕，正在准备 ASR 音频")
            audio_started = time.monotonic()
            audio_path = (
                download_audio_for_asr(url, out_dir, progress_callback=progress_callback)
                if progress_callback
                else download_audio_for_asr(url, out_dir)
            )
            _record_timing(timing_collector, "audio_download", audio_started)
            _validate_media_size(audio_path, settings)
        except DownloadError as exc:
            raise RuntimeError(str(exc)) from exc
        try:
            _progress(progress_callback, "正在进行语音识别 0.0%")
            asr_started = time.monotonic()
            adapter = get_asr_adapter(settings)
            raw_items = (
                adapter.transcribe(audio_path, progress_callback=progress_callback)
                if progress_callback
                else adapter.transcribe(audio_path)
            )
            _record_timing(timing_collector, "asr", asr_started)
            source = "asr"
        except ASRUnavailableError as exc:
            raise RuntimeError(f"asr_failed: {exc}") from exc
    if not raw_items:
        raise RuntimeError("subtitle_not_found: no subtitle entries and ASR returned no transcript.")
    segments = segment_transcript(raw_items)
    write_json(out_dir / "transcript.json", VideoTranscript(video_id=metadata.video_id, segments=segments, source=source))
    _progress(progress_callback, "正在提取关键帧和构建多模态证据")
    multimodal_started = time.monotonic()
    ensure_multimodal_segments(metadata.video_id, settings)
    _record_timing(timing_collector, "multimodal_preprocess", multimodal_started)
    return metadata.video_id


def _record_timing(timings: dict[str, float] | None, stage: str, started: float) -> None:
    if timings is not None:
        timings[stage] = round(max(0.0, time.monotonic() - started), 3)


def _validate_media_size(path: Path, settings: Settings) -> None:
    if path.exists() and path.stat().st_size > settings.max_download_bytes:
        raise RuntimeError(
            f"media_too_large: file size {path.stat().st_size} exceeds limit {settings.max_download_bytes}"
        )


def _cached_video_id_from_url(url: str, settings: Settings) -> str | None:
    candidate_ids = []
    youtube_id = _youtube_video_id_from_url(url)
    if youtube_id:
        candidate_ids.append(youtube_id)
    for video_id in candidate_ids:
        out_dir = video_dir(video_id, settings)
        if (out_dir / "metadata.json").exists() and (out_dir / "transcript.json").exists():
            return video_id
    videos_dir = settings.videos_dir
    if not videos_dir.exists():
        return None
    normalized_url = url.strip()
    for metadata_path in videos_dir.glob("*/metadata.json"):
        transcript_path = metadata_path.with_name("transcript.json")
        if not transcript_path.exists():
            continue
        try:
            metadata = VideoMetadata.model_validate(json.loads(metadata_path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
        if (metadata.url or "").strip() == normalized_url:
            return metadata.video_id
    return None


def _youtube_video_id_from_url(url: str) -> str | None:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().removeprefix("www.")
    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        return video_id if video_id and _looks_like_youtube_id(video_id) else None
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
        return video_id if video_id and _looks_like_youtube_id(video_id) else None
    match = re.search(r"(?:/shorts/|/embed/)([A-Za-z0-9_-]{6,})", parsed.path)
    if match and _looks_like_youtube_id(match.group(1)):
        return match.group(1)
    return None


def _looks_like_youtube_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{6,}", value))


def ensure_multimodal_segments(video_id: str, settings: Settings | None = None) -> list[MultimodalSegment]:
    settings = settings or get_settings()
    key = _cache_key(video_id, settings)
    with _CACHE_LOCK:
        cached = _RUN_MULTIMODAL_CACHE.get(key)
    if cached is not None:
        return cached
    out_dir = video_dir(video_id, settings)
    existing = out_dir / "multimodal_segments.json"
    with _CACHE_LOCK:
        rebuilt_this_run = key in _RUN_REBUILT_MULTIMODAL
    if existing.exists() and (not settings.force_refresh or rebuilt_this_run):
        logger.info("Reusing existing multimodal segments for video_id=%s", video_id)
        segments = load_multimodal_segments(existing)
        _sanitize_current_evidence_stage(segments)
        save_multimodal_segments(segments, out_dir)
        write_json(out_dir / "ocr.json", {"disabled": True, "reason": "OCR is disabled for the current evidence stage."})
        with _CACHE_LOCK:
            _RUN_MULTIMODAL_CACHE[key] = segments
        return segments
    metadata = read_metadata(video_id, settings)
    transcript_segments = read_segments(video_id, settings)
    logger.info("Building multimodal segments for video_id=%s", video_id)
    correlation_profile = _assess_video_correlation(video_id, metadata, transcript_segments, out_dir, settings)
    logger.info(
        "Correlation profile for video_id=%s: status=%s level=%s avg=%.3f frame_interval=%.1fs strategy=%s",
        video_id,
        correlation_profile.status,
        correlation_profile.relevance_level,
        correlation_profile.avg_confidence,
        correlation_profile.frame_interval,
        correlation_profile.representative_frame_strategy,
    )
    visual_profile = _classify_visual_profile(video_id, metadata, transcript_segments, out_dir, settings)
    logger.info(
        "Visual profile for video_id=%s: class=%s score=%.3f fps=%.2f max_per_segment=%s",
        video_id,
        visual_profile.visual_class,
        visual_profile.visual_strength_score,
        visual_profile.recommended_frame_fps,
        visual_profile.recommended_max_frames_per_segment,
    )
    frames = _candidate_frames_for_video(video_id, metadata, transcript_segments, out_dir, settings, visual_profile, correlation_profile)
    logger.info("Extracted %s candidate frame(s) before dedupe", len(frames))
    candidate_frames = dedupe_frames(frames) if frames else []
    if len(frames) >= 8 and len(candidate_frames) <= 1:
        logger.warning(
            "Frame dedupe reduced %s candidate frames to %s. This is suspicious for non-trivial videos; "
            "check dedupe threshold, cache reuse, and source video download.",
            len(frames),
            len(candidate_frames),
        )
    logger.info("Kept %s candidate frame(s) after dedupe", len(candidate_frames))
    max_per_segment = max(1, visual_profile.recommended_max_frames_per_segment)
    if correlation_profile.relevance_level == "low":
        max_per_segment = 1
    elif correlation_profile.relevance_level == "medium":
        max_per_segment = max(1, max_per_segment // 2)
    representative = select_representative_frames(
        transcript_segments,
        candidate_frames,
        max_per_segment=max_per_segment,
        min_representative_frames=max(1, visual_profile.recommended_min_representative_frames),
        min_global_gap_seconds=4.0 if visual_profile.visual_class == "strong_visual" else 12.0,
        strategy=correlation_profile.representative_frame_strategy,
    )
    representative_ids = {frame_id for ids in representative.values() for frame_id in ids}
    representative_frames = [frame for frame in candidate_frames if frame.frame_id in representative_ids]
    if metadata.source == "mock":
        captions = [FrameCaption(frame_id=frame.frame_id, timestamp=frame.timestamp, caption="", model="mock-no-vision") for frame in representative_frames]
    else:
        logger.info("Running VLM captions for %s representative frame(s); OCR is disabled", len(representative_frames))
        captions = VLMCaptioner(settings).caption_frames(representative_frames)
    multimodal_segments = build_multimodal_segments(transcript_segments, candidate_frames, representative, captions, ocr_results=None)
    write_json(out_dir / "frames.json", [frame.model_dump() for frame in frames])
    write_json(out_dir / "candidate_frames.json", [frame.model_dump() for frame in candidate_frames])
    save_visual_profile(visual_profile, out_dir)
    save_frame_captions(captions, out_dir)
    write_json(out_dir / "ocr.json", {"disabled": True, "reason": "OCR is disabled for the current evidence stage."})
    save_multimodal_segments(multimodal_segments, out_dir)
    with _CACHE_LOCK:
        _RUN_REBUILT_MULTIMODAL.add(key)
        _RUN_MULTIMODAL_CACHE[key] = multimodal_segments
    return multimodal_segments


def _sanitize_current_evidence_stage(segments: list[MultimodalSegment]) -> None:
    for segment in segments:
        segment.ocr_text = ""
        segment.evidence_types = [item for item in segment.evidence_types if item in {"speech", "frame", "frame_caption"}]
        text = float(segment.modality_weight.get("text", 1.0))
        vision = float(segment.modality_weight.get("vision", 0.0))
        total = text + vision
        segment.modality_weight = {"text": round(text / total, 3), "vision": round(vision / total, 3)} if total else {"text": 0.0, "vision": 0.0}


def _classify_visual_profile(
    video_id: str,
    metadata: VideoMetadata,
    segments: list[TranscriptSegment],
    out_dir: Path,
    settings: Settings,
) -> VideoVisualProfile:
    try:
        transcript_source = read_transcript_source(video_id, settings)
    except Exception:
        transcript_source = ""
    has_subtitles = transcript_source == "subtitle"
    if metadata.local_path and Path(metadata.local_path).exists():
        try:
            return classify_video_visual_intensity(video_id, metadata.local_path, out_dir, has_subtitles=has_subtitles, settings=settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Lightweight visual classifier failed; using weak-visual defaults: %s", exc)
    profile = VideoVisualProfile(
        video_id=video_id,
        video_path=str(metadata.local_path or ""),
        motion_score=0.0,
        visual_complexity_score=0.0,
        color_variation_score=0.0,
        has_subtitles=has_subtitles,
        visual_strength_score=0.0,
        visual_class="weak_visual",
        recommended_frame_fps=settings.weak_visual_frame_fps,
        recommended_max_frames_per_segment=settings.weak_visual_max_frames_per_segment,
        recommended_min_representative_frames=3,
        reason="no_local_video_or_classifier_failed",
    )
    save_visual_profile(profile, out_dir)
    return profile


def _assess_video_correlation(
    video_id: str,
    metadata: VideoMetadata,
    segments: list[TranscriptSegment],
    out_dir: Path,
    settings: Settings,
) -> VideoCorrelationProfile:
    try:
        return assess_video_text_correlation(
            video_id,
            segments,
            video_path=metadata.local_path,
            output_dir=out_dir,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Video/text correlation agent failed; preserving default multimodal strategy: %s", exc)
        profile = VideoCorrelationProfile(
            frame_confidences=[],
            avg_confidence=1.0,
            frame_interval=1.0,
            refine_frame_count=10,
            representative_frame_strategy="default",
            relevance_level="high",
            status="failed",
            correlation_available=False,
            fallback_reason="correlation_agent_failed",
            reason=str(exc).splitlines()[0][:500],
        )
        write_json(out_dir / "video_correlation.json", profile)
        return profile


def _candidate_frames_for_video(
    video_id: str,
    metadata: VideoMetadata,
    segments: list[TranscriptSegment],
    out_dir: Path,
    settings: Settings,
    visual_profile: VideoVisualProfile | None = None,
    correlation_profile: VideoCorrelationProfile | None = None,
) -> list[VideoFrame]:
    if metadata.local_path and Path(metadata.local_path).exists():
        try:
            fps = visual_profile.recommended_frame_fps if visual_profile else 1.0
            if correlation_profile and correlation_profile.relevance_level in {"low", "medium", "text_only"} and correlation_profile.frame_interval > 0:
                fps = min(fps, 1.0 / correlation_profile.frame_interval)
            return extract_frames(video_id, metadata.local_path, out_dir, fps=fps)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Frame extraction failed for local file; continuing without real frames: %s", exc)
    if metadata.source == "mock":
        return _mock_frames_for_segments(segments, out_dir)
    message = "No trusted local video file is available for frame extraction; not reusing old visual cache."
    if settings.strict_runtime:
        raise RuntimeError(message)
    logger.warning("%s Continuing without frame evidence.", message)
    return []


def _mock_frames_for_segments(segments: list[TranscriptSegment], out_dir: Path) -> list[VideoFrame]:
    frame_dir = out_dir / "frames" / "raw"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames: list[VideoFrame] = []
    for segment in segments:
        timestamp = segment.start
        frame_id = f"f_{int(timestamp * 1000):010d}"
        path = frame_dir / f"{frame_id}.jpg"
        if not path.exists():
            path.write_bytes(b"mock-frame")
        frames.append(VideoFrame(frame_id=frame_id, timestamp=timestamp, path=str(path), selected=True, selection_reason="mock_representative"))
    return frames


def summarize(video_id: str, settings: Settings | None = None) -> SummaryReport:
    settings = settings or get_settings()
    metadata = read_metadata(video_id, settings)
    segments = read_segments(video_id, settings)
    multimodal_segments = ensure_multimodal_segments(video_id, settings)
    profile = route_modality(metadata, segments)
    report = summarize_multimodal_video(metadata, multimodal_segments, profile, settings=settings, log_dir=video_dir(video_id, settings) / "llm_logs")
    out_dir = video_dir(video_id, settings)
    write_json(out_dir / "modality.json", profile)
    write_json(out_dir / "summary.json", report)
    return report


def build_storyline(video_id: str, query: str | None = None, settings: Settings | None = None) -> Storyline:
    settings = settings or get_settings()
    multimodal_segments = ensure_multimodal_segments(video_id, settings)
    storyline = (
        build_storyline_from_multimodal_segments(
            video_id,
            multimodal_segments,
            query=query,
            metadata=read_metadata(video_id, settings),
            settings=settings,
            log_dir=video_dir(video_id, settings) / "llm_logs",
        )
        if multimodal_segments
        else build_storyline_from_segments(video_id, read_segments(video_id, settings), query=query)
    )
    save_storyline(storyline, video_dir(video_id, settings))
    return storyline


def build_store(video_id: str, settings: Settings | None = None) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.add_segments(video_id, read_segments(video_id, settings))
    return store


def ask(video_id: str, question: str, settings: Settings | None = None):
    return chat(video_id, question, settings=settings)


def chat(
    video_id: str,
    question: str,
    conversation_id: str | None = None,
    settings: Settings | None = None,
    persist: bool = True,
    allow_refinement: bool = True,
    persist_trace: bool | None = None,
    persist_refinement: bool = True,
):
    settings = settings or get_settings()
    ensure_multimodal_segments(video_id, settings)
    turn = ConversationAgent(video_id, video_dir(video_id, settings), settings=settings).answer(
        question,
        conversation_id=conversation_id,
        persist=persist,
        allow_refinement=allow_refinement,
        persist_trace=persist_trace,
        persist_refinement=persist_refinement,
    )
    if persist:
        SQLiteStore(settings.data_dir / "vka.sqlite3").save_conversation_turn(turn)
    return turn


def refine_frames(video_id: str, start: float, end: float, question: str = "manual frame refinement", settings: Settings | None = None) -> list[MultimodalSegment]:
    settings = settings or get_settings()
    segments = ensure_multimodal_segments(video_id, settings)
    return FrameEvidenceRefiner(video_dir(video_id, settings), settings=settings).refine(video_id, question, (start, end), segments)


def partial_restart_for_evidence(
    video_id: str,
    target_ranges: list[RefineRange],
    question: str,
    settings: Settings | None = None,
    refinement_strategy: str = "local_first",
) -> PartialPipelineRestartResult:
    """Partially restart evidence acquisition for selected time ranges only."""
    settings = settings or get_settings()
    out_dir = video_dir(video_id, settings)
    segments = ensure_multimodal_segments(video_id, settings)
    transcript_segments = read_segments(video_id, settings)
    before_frames = {frame_id for segment in segments for frame_id in segment.representative_frame_ids}
    before_captions = {caption.frame_id for segment in segments for caption in segment.visual_captions if caption.caption.strip()}
    before_text_segments = {
        segment.segment_id
        for segment in segments
        if segment.transcript_text.strip() and any(_overlaps(segment.start, segment.end, item.start, item.end) for item in target_ranges)
    }
    refinement_id = uuid.uuid4().hex
    for range_index, item in enumerate(target_ranges[:3]):
        segments = _ensure_refined_text_segment(video_id, segments, transcript_segments, item)
        refiner = FrameEvidenceRefiner(out_dir, settings=settings)
        if "refinement_strategy" in inspect.signature(refiner.refine).parameters:
            segments = refiner.refine(
                video_id,
                question,
                (item.start, item.end),
                segments,
                refinement_strategy=refinement_strategy,
                refinement_id=refinement_id,
                range_index=range_index,
            )
        else:  # Compatibility for lightweight adapters and older test doubles.
            segments = refiner.refine(video_id, question, (item.start, item.end), segments)
    save_multimodal_segments(segments, out_dir)
    chroma_store = ChromaMemoryStore(settings)
    chroma_store.add_multimodal_segments(video_id, segments)
    after_frames = {frame_id for segment in segments for frame_id in segment.representative_frame_ids}
    after_captions = {caption.frame_id for segment in segments for caption in segment.visual_captions if caption.caption.strip()}
    after_text_segments = {
        segment.segment_id
        for segment in segments
        if segment.transcript_text.strip() and any(_overlaps(segment.start, segment.end, item.start, item.end) for item in target_ranges)
    }
    added_frame_ids = sorted(after_frames - before_frames)
    added_caption_ids = sorted(after_captions - before_captions)
    added_text_ids = sorted(after_text_segments - before_text_segments)
    added_evidence_ids = [f"{video_id}:{segment_id}:speech" for segment_id in added_text_ids]
    added_evidence_ids.extend(_frame_caption_evidence_id(video_id, segments, frame_id) for frame_id in added_caption_ids)
    return PartialPipelineRestartResult(
        ranges=target_ranges,
        added_frames=len(added_frame_ids),
        added_text_segments=len(added_text_ids),
        added_captions=len(added_caption_ids),
        added_evidence_ids=added_evidence_ids,
        status="updated" if added_frame_ids or added_caption_ids or added_text_ids else "no_new_evidence",
        refinement_strategy=refinement_strategy,
        refined_frame_count=len(added_frame_ids),
    )


def rebuild_index(video_id: str, settings: Settings | None = None) -> dict[str, object]:
    settings = settings or get_settings()
    segments = ensure_multimodal_segments(video_id, settings)
    store = ChromaMemoryStore(settings)
    return _index_multimodal_segments(video_id, segments, store)


def _index_multimodal_segments(video_id: str, segments: list[MultimodalSegment], store: ChromaMemoryStore | None = None) -> dict[str, object]:
    store = store or ChromaMemoryStore()
    store.add_multimodal_segments(video_id, segments)
    speech_docs = sum(1 for segment in segments if segment.transcript_text.strip())
    caption_docs = sum(1 for segment in segments for caption in segment.visual_captions if caption.caption.strip())
    return {"video_id": video_id, "using_chroma": store.using_chroma, "speech_documents": speech_docs, "frame_caption_documents": caption_docs}


def _can_resume_stage(
    state: ProcessingState,
    stage: str,
    settings: Settings,
    required_paths: list[Path] | None = None,
    fingerprint: str | None = None,
) -> bool:
    if settings.force_refresh:
        return False
    if settings.pipeline_force_stage and settings.pipeline_force_stage == stage:
        return False
    if not settings.pipeline_resume:
        return False
    return state.is_succeeded(stage, required_paths, fingerprint=fingerprint)


def _stage_fingerprint(stage: str, paths: list[Path], config: dict[str, object]) -> str:
    digest = hashlib.sha256(stage.encode("utf-8"))
    digest.update(json.dumps(config, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
    for path in paths:
        digest.update(str(path.name).encode("utf-8"))
        if not path.exists():
            digest.update(b"<missing>")
            continue
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _multimodal_fingerprint(out_dir: Path, settings: Settings) -> str:
    return _stage_fingerprint(
        "multimodal",
        [out_dir / "metadata.json", out_dir / "transcript.json"],
        {
            "vlm_model": settings.runtime_vlm_model,
            "visual_classifier_enabled": settings.visual_classifier_enabled,
            "visual_strength_threshold": settings.visual_strength_threshold,
            "weak_visual_frame_fps": settings.weak_visual_frame_fps,
            "strong_visual_frame_fps": settings.strong_visual_frame_fps,
            "weak_visual_max_frames": settings.weak_visual_max_frames_per_segment,
            "strong_visual_max_frames": settings.strong_visual_max_frames_per_segment,
        },
    )


def _index_fingerprint(out_dir: Path, settings: Settings) -> str:
    return _stage_fingerprint(
        "index",
        [out_dir / "multimodal_segments.json"],
        {"embedding_model": settings.embedding_model, "embedding_base_url": settings.runtime_embedding_base_url},
    )


def _summary_fingerprint(out_dir: Path, settings: Settings) -> str:
    return _stage_fingerprint(
        "summary",
        [out_dir / "multimodal_segments.json"],
        {
            "model": settings.runtime_llm_model,
            "max_segments": settings.summary_max_segments,
            "segment_chars": settings.summary_segment_chars,
            "caption_chars": settings.summary_caption_chars,
        },
    )


def _storyline_fingerprint(out_dir: Path, settings: Settings, query: str | None) -> str:
    return _stage_fingerprint(
        "storyline",
        [out_dir / "multimodal_segments.json"],
        {"model": settings.runtime_storyline_model, "top_k": settings.storyline_top_k, "query": query or ""},
    )


def _overlaps(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return max(start_a, start_b) <= min(end_a, end_b)


def _ensure_refined_text_segment(
    video_id: str,
    segments: list[MultimodalSegment],
    transcript_segments: list[TranscriptSegment],
    target_range: RefineRange,
) -> list[MultimodalSegment]:
    overlapping = [item for item in transcript_segments if _overlaps(item.start, item.end, target_range.start, target_range.end)]
    if not overlapping:
        return segments
    start = min(item.start for item in overlapping)
    end = max(item.end for item in overlapping)
    text = " ".join(item.text for item in overlapping if item.text.strip()).strip()
    if not text:
        return segments
    existing = [
        segment
        for segment in segments
        if _overlaps(segment.start, segment.end, start, end) and text in segment.transcript_text
    ]
    if existing:
        return segments
    segment_id = f"refined_text_{int(target_range.start)}_{int(target_range.end)}"
    if any(segment.segment_id == segment_id for segment in segments):
        return segments
    segments.append(
        MultimodalSegment(
            segment_id=segment_id,
            start=start,
            end=end,
            transcript_text=text,
            evidence_types=["speech"],
            modality_weight={"text": 1.0, "vision": 0.0},
        )
    )
    return sorted(segments, key=lambda item: (item.start, item.end, item.segment_id))


def _frame_caption_evidence_id(video_id: str, segments: list[MultimodalSegment], frame_id: str) -> str:
    for segment in segments:
        if any(caption.frame_id == frame_id for caption in segment.visual_captions):
            return f"{video_id}:{segment.segment_id}:frame_caption:{frame_id}"
    return f"{video_id}:unknown:frame_caption:{frame_id}"


def audit(video_id: str, settings: Settings | None = None) -> dict[str, object]:
    settings = settings or get_settings()
    out_dir = video_dir(video_id, settings)
    segments = ensure_multimodal_segments(video_id, settings)
    segment_ids = {segment.segment_id for segment in segments}
    errors: list[str] = []
    warnings: list[str] = []
    allowed = {"speech", "frame", "frame_caption"}
    for segment in segments:
        bad = [item for item in segment.evidence_types if item not in allowed]
        if bad:
            errors.append(f"segment {segment.segment_id} has invalid evidence_types={bad}")
        if segment.ocr_text:
            errors.append(f"segment {segment.segment_id} references OCR text while OCR is disabled")
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        summary = SummaryReport.model_validate(json.loads(summary_path.read_text(encoding="utf-8")))
        for item in summary.structured_outline:
            for segment_id in item.segment_ids:
                if segment_id not in segment_ids:
                    errors.append(f"summary references missing segment_id={segment_id}")
    storyline_path = out_dir / "storyline.json"
    if storyline_path.exists():
        storyline = load_storyline(storyline_path)
        for node in storyline.nodes:
            if node.status.value == "supported" and not node.evidence_segment_ids:
                errors.append(f"storyline node {node.node_id} is supported without evidence")
            for segment_id in node.evidence_segment_ids:
                if segment_id not in segment_ids:
                    errors.append(f"storyline references missing segment_id={segment_id}")
            if node.ocr_evidence:
                errors.append(f"storyline node {node.node_id} references OCR while disabled")
    qa_path = out_dir / "qa_history.json"
    if qa_path.exists():
        history = json.loads(qa_path.read_text(encoding="utf-8"))
        for item in history:
            bad = [value for value in item.get("evidence_types", []) if value not in allowed]
            if bad:
                errors.append(f"qa turn {item.get('turn_id')} has invalid evidence_types={bad}")
    if not summary_path.exists():
        warnings.append("summary.json is missing")
    if not storyline_path.exists():
        warnings.append("storyline.json is missing")
    return {"video_id": video_id, "ok": not errors, "errors": errors, "warnings": warnings}


def export_obsidian(video_id: str, settings: Settings | None = None) -> dict[str, Path]:
    settings = settings or get_settings()
    if not settings.obsidian_vault_path:
        raise RuntimeError("OBSIDIAN_VAULT_PATH is not configured.")
    metadata = read_metadata(video_id, settings)
    out_dir = video_dir(video_id, settings)
    summary_path = out_dir / "summary.json"
    storyline_path = out_dir / "storyline.json"
    summary = SummaryReport.model_validate(json.loads(summary_path.read_text(encoding="utf-8"))) if summary_path.exists() else summarize(video_id, settings)
    storyline = load_storyline(storyline_path) if storyline_path.exists() else build_storyline(video_id, settings=settings)
    multimodal_segments = ensure_multimodal_segments(video_id, settings)
    return write_obsidian_notes(settings.obsidian_vault_path, metadata, summary, storyline, multimodal_segments=multimodal_segments)


def _save_mempalace_memories(
    settings: Settings,
    metadata: VideoMetadata,
    report: SummaryReport,
    storyline: Storyline,
    multimodal_segments: list[MultimodalSegment],
) -> dict[str, object]:
    """Persist selected long-term memories without blocking the video pipeline."""
    stats: dict[str, object] = {"provider": settings.mempalace_provider, "model_summary": 0, "raw_evidence": 0, "status": "skipped"}
    if settings.mempalace_provider.lower() == "noop":
        return stats
    adapter = None
    try:
        from app.memory.knowledge_catalog import KnowledgeCatalog

        catalog = KnowledgeCatalog(settings)
        catalog.ingest_video(metadata, storyline, multimodal_segments)
        adapter = create_memory_adapter(settings)
        # Durable outbox retains failed writes; the core pipeline still completes.
        stats["knowledge"] = catalog.sync(adapter)
        with catalog.connect() as db:
            for encoded, in db.execute("SELECT payload FROM knowledge WHERE video_id=? AND active=1 AND synced=1", (metadata.video_id,)):
                kind = json.loads(encoded)["memory_type"]
                if kind in stats:
                    stats[kind] += 1
        stats["status"] = "pending" if stats["knowledge"]["pending"] else "saved"
    except Exception as exc:  # noqa: BLE001
        logger.warning("MemPalace memory save skipped: %s", exc)
        stats["status"] = "failed"
        stats["error"] = str(exc).splitlines()[0]
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
    return stats


def process(
    url: str | None = None,
    file_path: Path | None = None,
    query: str | None = None,
    export_to_obsidian: bool = False,
    mock: bool = False,
    settings: Settings | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    settings = (settings or get_settings()).model_copy(deep=True)
    settings.log_runtime_config()
    pipeline_started = time.monotonic()
    store = SQLiteStore(settings.data_dir / "vka.sqlite3")
    run_id = store.start_run(model_used=settings.llm_model)
    state: ProcessingState | None = None
    ingest_started = time.monotonic()
    ingest_timings: dict[str, float] = {}
    try:
        if mock:
            logger.info("Stage 1/7 ingest: mock")
            video_id = ingest_mock(settings)
        elif url:
            logger.info("Stage 1/7 ingest: url")
            video_id = ingest_url(
                url,
                settings,
                progress_callback=progress_callback,
                timing_collector=ingest_timings,
            )
        elif file_path:
            logger.info("Stage 1/7 ingest: local file")
            video_id = ingest_local_file(
                file_path,
                settings,
                progress_callback=progress_callback,
                timing_collector=ingest_timings,
            )
        else:
            raise RuntimeError("Provide --mock, --url, or --file.")
        out_dir = video_dir(video_id, settings)
        state = ProcessingState(out_dir)
        state.start("pipeline")
        ingest_duration = round(max(0.0, time.monotonic() - ingest_started), 3)
        state.succeed(
            "ingest",
            [out_dir / "metadata.json", out_dir / "transcript.json"],
            {"duration_seconds": ingest_duration},
        )
        for stage_name, duration in ingest_timings.items():
            state.succeed(stage_name, extra={"duration_seconds": duration})
        state.start("load_metadata_transcript")
        logger.info("Stage 2/7 load metadata/transcript")
        _progress(progress_callback, "正在加载元数据和转写文本")
        metadata = read_metadata(video_id, settings)
        llm_timeout_seconds = apply_dynamic_llm_timeout(settings, metadata)
        segments = read_segments(video_id, settings)
        state.succeed("load_metadata_transcript")
        logger.info("Stage 3/7 build multimodal index")
        _progress(progress_callback, "正在提取关键帧和构建多模态证据")
        multimodal_path = out_dir / "multimodal_segments.json"
        multimodal_fingerprint = _multimodal_fingerprint(out_dir, settings)
        if _can_resume_stage(state, "multimodal", settings, [multimodal_path], multimodal_fingerprint):
            multimodal_segments = load_multimodal_segments(multimodal_path)
            with _CACHE_LOCK:
                _RUN_MULTIMODAL_CACHE[_cache_key(video_id, settings)] = multimodal_segments
        else:
            state.start("multimodal")
            try:
                multimodal_segments = run_stage_with_restarts(
                    "multimodal",
                    lambda: ensure_multimodal_segments(video_id, settings),
                    state,
                    settings,
                )
                state.succeed(
                    "multimodal",
                    [multimodal_path, out_dir / "visual_profile.json", out_dir / "video_correlation.json", out_dir / "frame_captions.json"],
                    {"fingerprint": multimodal_fingerprint},
                )
            except Exception as exc:  # noqa: BLE001
                state.fail("multimodal", exc, retryable=True)
                raise
        _progress(progress_callback, "正在建立检索索引")
        chroma_store = ChromaMemoryStore(settings)
        index_fingerprint = _index_fingerprint(out_dir, settings)
        if _can_resume_stage(state, "index", settings, fingerprint=index_fingerprint) and chroma_store.has_video_documents(video_id):
            speech_docs = sum(1 for segment in multimodal_segments if segment.transcript_text.strip())
            caption_docs = sum(1 for segment in multimodal_segments for caption in segment.visual_captions if caption.caption.strip())
            index_stats = {"video_id": video_id, "using_chroma": chroma_store.using_chroma, "speech_documents": speech_docs, "frame_caption_documents": caption_docs, "resumed": True}
        else:
            state.start("index")
            try:
                index_stats = run_stage_with_restarts(
                    "index",
                    lambda: _index_multimodal_segments(video_id, multimodal_segments, chroma_store),
                    state,
                    settings,
                )
                state.succeed("index", extra={**index_stats, "fingerprint": index_fingerprint})
            except Exception as exc:  # noqa: BLE001
                state.fail("index", exc, retryable=True)
                raise
        profile: ModalityProfile = route_modality(metadata, segments)
        write_json(out_dir / "modality.json", profile)
        logger.info("Stage 4-5/7 summarize and build storyline")
        summary_path = out_dir / "summary.json"
        storyline_path = out_dir / "storyline.json"
        report: SummaryReport | None = None
        storyline: Storyline | None = None
        summary_fingerprint = _summary_fingerprint(out_dir, settings)
        storyline_fingerprint = _storyline_fingerprint(out_dir, settings, query)
        need_summary = not _can_resume_stage(state, "summary", settings, [summary_path], summary_fingerprint)
        need_storyline = not _can_resume_stage(state, "storyline", settings, [storyline_path], storyline_fingerprint)
        if not need_summary:
            report = SummaryReport.model_validate(json.loads(summary_path.read_text(encoding="utf-8")))
        if not need_storyline:
            storyline = load_storyline(storyline_path)
        if need_summary or need_storyline:
            if need_summary:
                _progress(progress_callback, "正在生成摘要")
                state.start("summary")
                try:
                    report = run_stage_with_restarts(
                        "summary",
                        lambda: summarize_multimodal_video(metadata, multimodal_segments, profile, settings, out_dir / "llm_logs"),
                        state,
                        settings,
                    )
                    write_json(summary_path, report)
                    state.succeed(
                        "summary",
                        [summary_path],
                        {"generation_status": report.generation_status, "fingerprint": summary_fingerprint},
                    )
                except Exception as exc:  # noqa: BLE001
                    state.fail("summary", exc, retryable=True)
                    raise
            assert report is not None
            # Summary/global context supplies canonical entities; replacements are
            # applied locally and the raw ASR remains preserved for auditability.
            multimodal_segments, entities = normalize_segments(
                multimodal_segments,
                metadata,
                report,
                output_path=out_dir / "entity_normalization.json",
                auto_apply_threshold=settings.entity_normalization_apply_threshold,
            )
            report = report.model_copy(update={"canonical_entities": entities})
            save_multimodal_segments(multimodal_segments, out_dir)
            write_json(summary_path, report)
            # The initial index may have been built before summary generation; always
            # synchronize it after normalization so Chroma/BM25 consume normalized text.
            index_stats = _index_multimodal_segments(video_id, multimodal_segments, chroma_store)
            if need_storyline:
                _progress(progress_callback, "正在生成 storyline")
                state.start("storyline")
                try:
                    storyline = run_stage_with_restarts(
                        "storyline",
                        lambda: build_storyline_from_multimodal_segments(video_id, multimodal_segments, query, metadata, settings, out_dir / "llm_logs"),
                        state,
                        settings,
                    )
                    save_storyline(storyline, out_dir)
                    state.succeed("storyline", [storyline_path], {"fingerprint": storyline_fingerprint})
                except Exception as exc:  # noqa: BLE001
                    state.fail("storyline", exc, retryable=True)
                    raise
        # Resume-safe normalization: legacy runs may have a summary/storyline but
        # no normalized transcript or entity trace yet.
        if report is not None and (
            not (out_dir / "entity_normalization.json").exists()
            or any(not segment.normalized_transcript_text for segment in multimodal_segments)
        ):
            multimodal_segments, entities = normalize_segments(
                multimodal_segments,
                metadata,
                report,
                output_path=out_dir / "entity_normalization.json",
                auto_apply_threshold=settings.entity_normalization_apply_threshold,
            )
            report = report.model_copy(update={"canonical_entities": entities})
            save_multimodal_segments(multimodal_segments, out_dir)
            write_json(summary_path, report)
            index_stats = _index_multimodal_segments(video_id, multimodal_segments, chroma_store)
        assert report is not None
        assert storyline is not None
        logger.info("Stage 6/7 persist")
        _progress(progress_callback, "正在保存处理结果")
        state.start("persist")
        store.save_video(metadata)
        store.save_segments(video_id, segments)
        store.save_summary(report)
        store.save_storyline(storyline)
        memory_stats = _save_mempalace_memories(settings, metadata, report, storyline, multimodal_segments)
        state.succeed("persist", extra={"memory": memory_stats})
        logger.info("Stage 7/7 export")
        _progress(progress_callback, "正在完成处理")
        obsidian_paths = {}
        if export_to_obsidian:
            state.start("export_obsidian")
            try:
                obsidian_paths = run_stage_with_restarts(
                    "export_obsidian",
                    lambda: export_obsidian(video_id, settings),
                    state,
                    settings,
                )
                state.succeed("export_obsidian", [Path(value) for value in obsidian_paths.values() if isinstance(value, Path)])
            except Exception as exc:  # noqa: BLE001
                state.fail("export_obsidian", exc, retryable=True)
                raise
        state.succeed("pipeline", extra={"total_duration_seconds": round(time.monotonic() - pipeline_started, 3)})
        store.finish_run(run_id, "succeeded", modality_mode=profile.mode.value)
        return {
            "video_id": video_id,
            "summary": report,
            "storyline": storyline,
            "multimodal_segments": multimodal_segments,
            "index_stats": index_stats,
            "obsidian": obsidian_paths,
            "stage_timings": state.timings(),
            "processing_state_path": str(state.path),
            "memory": memory_stats,
            "llm_timeout_seconds": llm_timeout_seconds,
        }
    except Exception as exc:  # noqa: BLE001
        if state is not None:
            state.fail("pipeline", exc, retryable=True)
        store.finish_run(run_id, "failed", errors=str(exc))
        logger.error("Pipeline failed: %s", exc)
        raise


def process_existing(
    video_id: str,
    query: str | None = None,
    export_to_obsidian: bool = False,
    settings: Settings | None = None,
) -> dict[str, object]:
    """Run the post-ingest pipeline for an existing video directory.

    This is used by streaming/batch workers after stage-one ingest has already
    produced metadata.json and transcript.json. It intentionally mirrors the
    post-ingest stages of process() without re-fetching the original source.
    """
    settings = (settings or get_settings()).model_copy(deep=True)
    settings.log_runtime_config()
    pipeline_started = time.monotonic()
    store = SQLiteStore(settings.data_dir / "vka.sqlite3")
    run_id = store.start_run(video_id=video_id, model_used=settings.llm_model)
    state: ProcessingState | None = None
    try:
        out_dir = video_dir(video_id, settings)
        if not (out_dir / "metadata.json").exists() or not (out_dir / "transcript.json").exists():
            raise RuntimeError(f"video_not_ingested: missing metadata.json or transcript.json for video_id={video_id}")
        state = ProcessingState(out_dir)
        state.start("pipeline")
        state.succeed("ingest", [out_dir / "metadata.json", out_dir / "transcript.json"])
        state.start("load_metadata_transcript")
        logger.info("Stage 2/7 load metadata/transcript")
        metadata = read_metadata(video_id, settings)
        llm_timeout_seconds = apply_dynamic_llm_timeout(settings, metadata)
        segments = read_segments(video_id, settings)
        state.succeed("load_metadata_transcript")
        logger.info("Stage 3/7 build multimodal index")
        multimodal_path = out_dir / "multimodal_segments.json"
        multimodal_fingerprint = _multimodal_fingerprint(out_dir, settings)
        if _can_resume_stage(state, "multimodal", settings, [multimodal_path], multimodal_fingerprint):
            multimodal_segments = load_multimodal_segments(multimodal_path)
            with _CACHE_LOCK:
                _RUN_MULTIMODAL_CACHE[_cache_key(video_id, settings)] = multimodal_segments
        else:
            state.start("multimodal")
            multimodal_segments = run_stage_with_restarts(
                "multimodal",
                lambda: ensure_multimodal_segments(video_id, settings),
                state,
                settings,
            )
            state.succeed(
                "multimodal",
                [multimodal_path, out_dir / "visual_profile.json", out_dir / "video_correlation.json", out_dir / "frame_captions.json"],
                {"fingerprint": multimodal_fingerprint},
            )
        chroma_store = ChromaMemoryStore(settings)
        index_fingerprint = _index_fingerprint(out_dir, settings)
        if _can_resume_stage(state, "index", settings, fingerprint=index_fingerprint) and chroma_store.has_video_documents(video_id):
            speech_docs = sum(1 for segment in multimodal_segments if segment.transcript_text.strip())
            caption_docs = sum(1 for segment in multimodal_segments for caption in segment.visual_captions if caption.caption.strip())
            index_stats = {"video_id": video_id, "using_chroma": chroma_store.using_chroma, "speech_documents": speech_docs, "frame_caption_documents": caption_docs, "resumed": True}
        else:
            state.start("index")
            index_stats = run_stage_with_restarts(
                "index",
                lambda: _index_multimodal_segments(video_id, multimodal_segments, chroma_store),
                state,
                settings,
            )
            state.succeed("index", extra={**index_stats, "fingerprint": index_fingerprint})
        profile: ModalityProfile = route_modality(metadata, segments)
        write_json(out_dir / "modality.json", profile)
        logger.info("Stage 4-5/7 summarize and build storyline")
        summary_path = out_dir / "summary.json"
        storyline_path = out_dir / "storyline.json"
        report: SummaryReport | None = None
        storyline: Storyline | None = None
        summary_fingerprint = _summary_fingerprint(out_dir, settings)
        storyline_fingerprint = _storyline_fingerprint(out_dir, settings, query)
        need_summary = not _can_resume_stage(state, "summary", settings, [summary_path], summary_fingerprint)
        need_storyline = not _can_resume_stage(state, "storyline", settings, [storyline_path], storyline_fingerprint)
        if not need_summary:
            report = SummaryReport.model_validate(json.loads(summary_path.read_text(encoding="utf-8")))
        if not need_storyline:
            storyline = load_storyline(storyline_path)
        if need_summary or need_storyline:
            if need_summary:
                state.start("summary")
                report = run_stage_with_restarts(
                    "summary",
                    lambda: summarize_multimodal_video(metadata, multimodal_segments, profile, settings, out_dir / "llm_logs"),
                    state,
                    settings,
                )
                write_json(summary_path, report)
                state.succeed(
                    "summary",
                    [summary_path],
                    {"generation_status": report.generation_status, "fingerprint": summary_fingerprint},
                )
            assert report is not None
            multimodal_segments, entities = normalize_segments(
                multimodal_segments,
                metadata,
                report,
                output_path=out_dir / "entity_normalization.json",
                auto_apply_threshold=settings.entity_normalization_apply_threshold,
            )
            report = report.model_copy(update={"canonical_entities": entities})
            save_multimodal_segments(multimodal_segments, out_dir)
            write_json(summary_path, report)
            index_stats = _index_multimodal_segments(video_id, multimodal_segments, chroma_store)
            if need_storyline:
                state.start("storyline")
                storyline = run_stage_with_restarts(
                    "storyline",
                    lambda: build_storyline_from_multimodal_segments(video_id, multimodal_segments, query, metadata, settings, out_dir / "llm_logs"),
                    state,
                    settings,
                )
                save_storyline(storyline, out_dir)
                state.succeed("storyline", [storyline_path], {"fingerprint": storyline_fingerprint})
        if report is not None and (
            not (out_dir / "entity_normalization.json").exists()
            or any(not segment.normalized_transcript_text for segment in multimodal_segments)
        ):
            multimodal_segments, entities = normalize_segments(
                multimodal_segments,
                metadata,
                report,
                output_path=out_dir / "entity_normalization.json",
                auto_apply_threshold=settings.entity_normalization_apply_threshold,
            )
            report = report.model_copy(update={"canonical_entities": entities})
            save_multimodal_segments(multimodal_segments, out_dir)
            write_json(summary_path, report)
            index_stats = _index_multimodal_segments(video_id, multimodal_segments, chroma_store)
        assert report is not None
        assert storyline is not None
        logger.info("Stage 6/7 persist")
        store.save_video(metadata)
        store.save_segments(video_id, segments)
        store.save_summary(report)
        store.save_storyline(storyline)
        memory_stats = _save_mempalace_memories(settings, metadata, report, storyline, multimodal_segments)
        state.succeed("persist", extra={"memory": memory_stats})
        logger.info("Stage 7/7 export")
        obsidian_paths = {}
        if export_to_obsidian:
            state.start("export_obsidian")
            obsidian_paths = run_stage_with_restarts(
                "export_obsidian",
                lambda: export_obsidian(video_id, settings),
                state,
                settings,
            )
            state.succeed("export_obsidian", [Path(value) for value in obsidian_paths.values() if isinstance(value, Path)])
        state.succeed("pipeline", extra={"total_duration_seconds": round(time.monotonic() - pipeline_started, 3)})
        store.finish_run(run_id, "succeeded", modality_mode=profile.mode.value)
        return {
            "video_id": video_id,
            "summary": report,
            "storyline": storyline,
            "multimodal_segments": multimodal_segments,
            "index_stats": index_stats,
            "obsidian": obsidian_paths,
            "stage_timings": state.timings(),
            "processing_state_path": str(state.path),
            "memory": memory_stats,
            "llm_timeout_seconds": llm_timeout_seconds,
        }
    except Exception as exc:  # noqa: BLE001
        if state is not None:
            state.fail("pipeline", exc, retryable=True)
        store.finish_run(run_id, "failed", errors=str(exc))
        logger.error("Post-ingest pipeline failed: %s", exc)
        raise


def _preflight_url_runtime(settings: Settings) -> None:
    settings.validate_runtime_config()
    local_ffmpeg = Path("tools/ffmpeg/bin/ffmpeg.exe").resolve()
    local_ffprobe = Path("tools/ffmpeg/bin/ffprobe.exe").resolve()
    if local_ffmpeg.exists() and local_ffprobe.exists():
        os.environ["PATH"] = f"{local_ffmpeg.parent}{os.pathsep}{os.environ.get('PATH', '')}"
        return
    if shutil.which("ffmpeg") is None:
        message = "ffmpeg is required for yt-dlp video/audio merge. Install ffmpeg and ensure it is in PATH."
        if settings.strict_runtime:
            raise RuntimeError(message)
        logger.warning("%s Continuing in degraded mode.", message)
