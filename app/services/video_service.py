from __future__ import annotations

import json
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, get_settings
from app import pipeline
from app.models import ModalityProfile, MultimodalSegment, Storyline, SummaryReport, VideoMetadata

SUPPORTED_UPLOAD_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".mp3", ".wav", ".m4a"}
_SOURCE_LOCKS: dict[str, threading.Lock] = {}
_SOURCE_LOCKS_GUARD = threading.Lock()


@dataclass
class ProcessVideoResult:
    success: bool
    video_id: str | None = None
    metadata: VideoMetadata | None = None
    summary: SummaryReport | None = None
    storyline: Storyline | None = None
    modality_profile: ModalityProfile | None = None
    suggested_questions: list[str] | None = None
    obsidian_path: str | None = None
    obsidian_status: str = ""
    multimodal_segments: list[MultimodalSegment] | None = None
    processing_state: dict | None = None
    error: str | None = None


class VideoService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def process_video(
        self,
        url: str | None,
        uploaded_file_path: str | None,
        query: str | None = None,
        progress_callback=None,
    ) -> ProcessVideoResult:
        url = (url or "").strip()
        uploaded_file_path = (uploaded_file_path or "").strip()
        if not url and not uploaded_file_path:
            return ProcessVideoResult(success=False, error="请先输入视频 URL 或上传本地视频/音频文件。")

        try:
            source_key = str(Path(uploaded_file_path).resolve()) if uploaded_file_path else url
            with _source_lock(source_key):
                if uploaded_file_path:
                    _progress(progress_callback, "正在复制上传文件")
                    file_path = self.copy_upload_to_data(uploaded_file_path)
                    result = self._run_pipeline(file_path=file_path, query=query, progress_callback=progress_callback)
                else:
                    _progress(progress_callback, "正在解析视频 URL")
                    result = self._run_pipeline(url=url, query=query, progress_callback=progress_callback)
            return result
        except Exception as exc:  # noqa: BLE001
            return ProcessVideoResult(success=False, error=_user_error(exc))

    def load_latest_completed(self) -> ProcessVideoResult | None:
        """Restore the newest fully processed video after a UI/server restart."""
        if not self.settings.videos_dir.exists():
            return None
        state_paths = sorted(
            self.settings.videos_dir.glob("*/processing_state.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for state_path in state_paths:
            try:
                processing_state = json.loads(state_path.read_text(encoding="utf-8"))
                if processing_state.get("stages", {}).get("pipeline", {}).get("status") != "succeeded":
                    continue
                video_id = state_path.parent.name
                metadata = pipeline.read_metadata(video_id, self.settings)
                summary = SummaryReport.model_validate_json((state_path.parent / "summary.json").read_text(encoding="utf-8"))
                storyline = pipeline.load_storyline(state_path.parent / "storyline.json")
                multimodal_path = state_path.parent / "multimodal_segments.json"
                multimodal_segments = pipeline.load_multimodal_segments(multimodal_path) if multimodal_path.exists() else []
                _apply_outline_time_ranges(summary, multimodal_segments)
                return ProcessVideoResult(
                    success=True,
                    video_id=video_id,
                    metadata=metadata,
                    summary=summary,
                    storyline=storyline,
                    modality_profile=self._read_modality(video_id),
                    suggested_questions=summary.open_questions,
                    obsidian_status="已从磁盘恢复最近完成的视频上下文。",
                    multimodal_segments=multimodal_segments,
                    processing_state=processing_state,
                )
            except (OSError, ValueError, KeyError):
                continue
        return None

    def copy_upload_to_data(self, uploaded_file_path: str) -> Path:
        source = Path(uploaded_file_path)
        if not source.exists():
            raise ValueError(f"上传文件不存在：{source}")
        if source.suffix.lower() not in SUPPORTED_UPLOAD_SUFFIXES:
            raise ValueError(f"不支持的文件格式：{source.suffix}。支持：{', '.join(sorted(SUPPORTED_UPLOAD_SUFFIXES))}")
        if source.stat().st_size > self.settings.max_download_bytes:
            raise ValueError(f"上传文件过大：{source.stat().st_size} bytes，限制为 {self.settings.max_download_bytes} bytes")
        upload_dir = self.settings.data_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = _unique_path(upload_dir / safe_filename(source.name))
        shutil.copy2(source, target)
        return target

    def _run_pipeline(
        self,
        url: str | None = None,
        file_path: Path | None = None,
        query: str | None = None,
        progress_callback=None,
    ) -> ProcessVideoResult:
        _progress(progress_callback, "正在处理视频：字幕 / ASR / 抽帧 / 摘要 / storyline")
        result = pipeline.process(
            url=url,
            file_path=file_path,
            query=query,
            export_to_obsidian=False,
            settings=self.settings,
            progress_callback=progress_callback,
        )
        video_id = str(result["video_id"])
        metadata = pipeline.read_metadata(video_id, self.settings)
        summary = result["summary"] if isinstance(result.get("summary"), SummaryReport) else pipeline.summarize(video_id, self.settings)
        storyline = result["storyline"] if isinstance(result.get("storyline"), Storyline) else pipeline.build_storyline(video_id, query, self.settings)
        modality_profile = self._read_modality(video_id)
        processing_state = self._read_processing_state(video_id)
        multimodal_segments = result.get("multimodal_segments") if isinstance(result.get("multimodal_segments"), list) else []
        _apply_outline_time_ranges(summary, multimodal_segments)
        _progress(progress_callback, "正在准备处理结果")
        obsidian_status = "未配置 Obsidian vault，已跳过导出。"
        obsidian_path = None
        if self.settings.obsidian_vault_path:
            try:
                _progress(progress_callback, "正在导出 Obsidian")
                paths = pipeline.export_obsidian(video_id, self.settings)
                obsidian_path = ", ".join(str(path) for path in paths.values())
                obsidian_status = "已导出 Obsidian。"
            except Exception as exc:  # noqa: BLE001
                obsidian_status = f"Obsidian 导出失败：{_user_error(exc)}"
        _progress(progress_callback, "处理完成")
        return ProcessVideoResult(
            success=True,
            video_id=video_id,
            metadata=metadata,
            summary=summary,
            storyline=storyline,
            modality_profile=modality_profile,
            suggested_questions=summary.open_questions,
            obsidian_path=obsidian_path,
            obsidian_status=obsidian_status,
            multimodal_segments=multimodal_segments,
            processing_state=processing_state,
        )

    def _read_modality(self, video_id: str) -> ModalityProfile | None:
        path = pipeline.video_dir(video_id, self.settings) / "modality.json"
        if not path.exists():
            return None
        return ModalityProfile.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def _read_processing_state(self, video_id: str) -> dict | None:
        path = pipeline.video_dir(video_id, self.settings) / "processing_state.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None


def safe_filename(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "-", value).strip(" .-")
    return cleaned or "upload"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"无法生成上传文件名：{path}")


def _progress(callback, message: str) -> None:
    if callback:
        callback(message)


def _user_error(exc: Exception) -> str:
    text = str(exc)
    if "OPENAI_API_KEY" in text or "LLM config" in text:
        return "LLM API key 未配置或不可用。"
    if "embedding_unavailable" in text:
        return "Embedding 不可用，请检查 EMBEDDING_MODEL / API 配置。"
    return text.splitlines()[0][:1000]


def _source_lock(source_key: str) -> threading.Lock:
    with _SOURCE_LOCKS_GUARD:
        return _SOURCE_LOCKS.setdefault(source_key, threading.Lock())


def _apply_outline_time_ranges(summary: SummaryReport, segments: list[MultimodalSegment]) -> None:
    by_id = {segment.segment_id: segment for segment in segments}
    for item in summary.structured_outline:
        cited = [by_id[segment_id] for segment_id in item.segment_ids if segment_id in by_id]
        if not cited:
            continue
        start = min(segment.start for segment in cited)
        end = max(segment.end for segment in cited)
        item.timestamp = f"{_timestamp(start)} - {_timestamp(end)}"


def _timestamp(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
