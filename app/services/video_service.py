from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, get_settings
from app import pipeline
from app.models import ModalityProfile, MultimodalSegment, Storyline, SummaryReport, VideoMetadata

SUPPORTED_UPLOAD_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".mp3", ".wav", ".m4a"}


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

    def copy_upload_to_data(self, uploaded_file_path: str) -> Path:
        source = Path(uploaded_file_path)
        if not source.exists():
            raise ValueError(f"上传文件不存在：{source}")
        if source.suffix.lower() not in SUPPORTED_UPLOAD_SUFFIXES:
            raise ValueError(f"不支持的文件格式：{source.suffix}。支持：{', '.join(sorted(SUPPORTED_UPLOAD_SUFFIXES))}")
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
        result = pipeline.process(url=url, file_path=file_path, query=query, export_to_obsidian=False, settings=self.settings)
        video_id = str(result["video_id"])
        metadata = pipeline.read_metadata(video_id, self.settings)
        summary = result["summary"] if isinstance(result.get("summary"), SummaryReport) else pipeline.summarize(video_id, self.settings)
        storyline = result["storyline"] if isinstance(result.get("storyline"), Storyline) else pipeline.build_storyline(video_id, query, self.settings)
        modality_profile = self._read_modality(video_id)
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
            multimodal_segments=result.get("multimodal_segments") if isinstance(result.get("multimodal_segments"), list) else None,
        )

    def _read_modality(self, video_id: str) -> ModalityProfile | None:
        path = pipeline.video_dir(video_id, self.settings) / "modality.json"
        if not path.exists():
            return None
        return ModalityProfile.model_validate(json.loads(path.read_text(encoding="utf-8")))


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
