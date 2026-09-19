from app.ui.gradio_app import _diagnostics, _gallery_from_evidence, _progress_html, _progress_value


def test_ui_progress_covers_pipeline_stages():
    stages = ["解析 URL", "字幕 ASR", "抽帧", "多模态", "摘要", "storyline", "检索", "Obsidian", "完成"]
    values = [_progress_value(stage) for stage in stages]
    assert max(values) >= 95
    assert values[-1] == 100


def test_ui_maps_live_download_and_asr_progress():
    assert _progress_value("正在下载视频 50.0% · 1.0 MiB/s") == 19
    assert _progress_value("正在下载音频 50.0%") == 34
    assert _progress_value("正在进行语音识别 50.0%") == 46
    html = _progress_html(46, "正在进行语音识别")
    assert "aria-valuenow='46'" in html
    assert "width:46%" in html


def test_ui_gallery_uses_evidence_image_paths():
    paths = _gallery_from_evidence([{"image_path": "a.jpg"}, {"image_path": "a.jpg"}, {"image_path": "b.jpg"}])
    assert paths == ["a.jpg", "b.jpg"]


def test_ui_gallery_does_not_drop_late_evidence_frames():
    evidence = [{"image_path": f"frame_{index}.jpg"} for index in range(20)]
    assert len(_gallery_from_evidence(evidence)) == 20


def test_diagnostics_reports_unsupported_claims():
    class Result:
        success = True
        summary = None
        storyline = None

        def __init__(self):
            self.storyline = {
                "nodes": [
                    {"node_id": "n1", "status": "unsupported", "evidence_segment_ids": []},
                    {"node_id": "n2", "status": "supported", "evidence_segment_ids": ["seg_1"]},
                ]
            }

    diagnostics = _diagnostics(Result())
    assert diagnostics["unsupported_claims"] == ["n1"]
    assert diagnostics["evidence_missing"] == ["n1"]
