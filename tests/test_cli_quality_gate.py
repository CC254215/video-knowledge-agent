from typer.testing import CliRunner

from app.cli import app


def test_quality_gate_returns_structured_success(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr("app.pipeline.refine_frames", lambda video_id, start, end, question="": [object()])
    monkeypatch.setattr("app.pipeline.audit", lambda video_id: {"video_id": video_id, "ok": True, "errors": [], "warnings": []})
    monkeypatch.setattr("app.cli._setup_logging", lambda: None)
    monkeypatch.setattr("app.cli._config_payload", lambda: {"has_llm_config": True})

    result = runner.invoke(app, ["quality-gate", "--video-id", "v1", "--start", "0", "--end", "1"])
    assert result.exit_code == 0
    assert '"status": "success"' in result.output
    assert "refine_frames" in result.output


def test_quality_gate_failure_exit_code(monkeypatch):
    runner = CliRunner()

    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.pipeline.refine_frames", fail)
    monkeypatch.setattr("app.pipeline.audit", lambda video_id: {"video_id": video_id, "ok": True, "errors": [], "warnings": []})
    monkeypatch.setattr("app.cli._setup_logging", lambda: None)
    monkeypatch.setattr("app.cli._config_payload", lambda: {"has_llm_config": True})

    result = runner.invoke(app, ["quality-gate", "--video-id", "v1"])
    assert result.exit_code == 1
    assert '"status": "failure"' in result.output
