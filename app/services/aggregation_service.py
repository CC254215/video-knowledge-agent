from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.analysis.multi_video_aggregator import MultiVideoAggregator
from app.config import Settings, get_settings


@dataclass
class AggregationResult:
    success: bool
    message: str
    current_video_id: str | None = None
    current_video_included: bool = False
    topic_count: int = 0
    record_count: int = 0
    obsidian_paths: list[str] | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None


class AggregationService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def aggregate_current_video(self, current_video_id: str | None) -> AggregationResult:
        if not current_video_id:
            return AggregationResult(success=False, message="请先处理或选择一个视频。", error="missing_current_video")
        try:
            result = MultiVideoAggregator(self.settings).run(video_ids=None, export_obsidian=bool(self.settings.obsidian_vault_path))
            if self.settings.mempalace_provider != "noop":
                from app.memory.knowledge_catalog import KnowledgeCatalog

                try:
                    catalog = KnowledgeCatalog(self.settings)
                    result["memory_topics"] = catalog.topics(current_video_id)
                    result["memory_catalog"] = catalog.stats()
                except Exception as exc:
                    result["memory_warning"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            return AggregationResult(success=False, message=f"聚合失败：{exc}", current_video_id=current_video_id, error=str(exc))
        topics = result.get("topics") if isinstance(result.get("topics"), list) else []
        included = any(current_video_id in (topic.get("videos") or []) for topic in topics if isinstance(topic, dict))
        message = (
            f"聚合完成：{result.get('record_count', 0)} 条观点记录，"
            f"{result.get('topic_count', 0)} 个跨视频主题。"
        )
        if not included:
            message += " 当前视频尚未进入聚合结果，请确认它已有 storyline.json 和 multimodal_segments.json。"
        return AggregationResult(
            success=True,
            message=message,
            current_video_id=current_video_id,
            current_video_included=included,
            topic_count=int(result.get("topic_count") or 0),
            record_count=int(result.get("record_count") or 0),
            obsidian_paths=[str(path) for path in result.get("obsidian_paths", [])],
            raw=result,
        )
