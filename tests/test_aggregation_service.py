from __future__ import annotations

from app.config import Settings
from app.services.aggregation_service import AggregationService


def test_aggregation_service_reports_current_video_included(monkeypatch, tmp_path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path, mempalace_provider="noop")

    def fake_run(self, video_ids=None, k=None, export_obsidian=True):
        return {
            "record_count": 3,
            "topic_count": 1,
            "topics": [{"videos": ["v1"], "cluster_id": 0}],
            "obsidian_paths": [],
        }

    monkeypatch.setattr("app.services.aggregation_service.MultiVideoAggregator.run", fake_run)

    result = AggregationService(settings).aggregate_current_video("v1")

    assert result.success
    assert result.current_video_included
    assert result.topic_count == 1
    assert result.record_count == 3


def test_aggregation_service_requires_current_video(tmp_path) -> None:
    result = AggregationService(Settings(_env_file=None, data_dir=tmp_path)).aggregate_current_video(None)

    assert not result.success
    assert result.error == "missing_current_video"
