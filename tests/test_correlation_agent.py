from pathlib import Path

import pytest

from app.config import Settings
from app.models import TranscriptSegment, VideoFrame
from app.vision import correlation_agent


def test_correlation_prompt_contains_video_type_instruction():
    assert "【第一步】视频类型判断" in correlation_agent.CORRELATION_PROMPT
    assert "视频类型：<判断结果>" in correlation_agent.CORRELATION_PROMPT
    assert "置信度：<0.0～1.0" in correlation_agent.CORRELATION_PROMPT


def test_correlation_text_too_short_defaults_high(tmp_path: Path):
    profile = correlation_agent.assess_video_text_correlation(
        "v1",
        [TranscriptSegment(segment_id="s0", start=0, end=1, text="短")],
        video_path=None,
        output_dir=tmp_path,
        settings=Settings(_env_file=None, correlation_text_min_chars=10),
    )

    assert profile.status == "text_too_short"
    assert profile.relevance_level == "high"
    assert profile.avg_confidence == 1.0


def test_correlation_video_unavailable_uses_text_only(tmp_path: Path):
    profile = correlation_agent.assess_video_text_correlation(
        "v1",
        [TranscriptSegment(segment_id="s0", start=0, end=10, text="这是一段足够长的台词文本，用来测试视频不可用时只处理文字相关性。")],
        video_path=tmp_path / "missing.mp4",
        output_dir=tmp_path,
        settings=Settings(_env_file=None, correlation_text_min_chars=5),
    )

    assert profile.status == "video_unavailable"
    assert profile.relevance_level == "text_only"
    assert (tmp_path / "video_correlation.json").exists()


def test_parse_batch_confidences_and_profile_strategy():
    values = correlation_agent._parse_confidences(
        """
视频类型：新闻
评分依据：新闻画面强对应
输入图片：记者
输入文本：洪水
场景合理性：1.0
内容匹配度：0.9
置信度：0.9
输入图片：跑鞋
输入文本：洪水
场景合理性：0.1
内容匹配度：0.0
置信度：0.1
"""
    )

    assert values == [0.9, 0.1]


def test_batch_scoring_uses_supplied_prompt(monkeypatch, tmp_path: Path):
    image = tmp_path / "f0.jpg"
    image.write_bytes(b"fake-image")
    frame = VideoFrame(frame_id="f0", timestamp=1.0, path=str(image))
    settings = Settings(
        _env_file=None,
        llm_correlation_api_key="key",
        llm_correlation_base_url="https://example.test/v4",
        llm_correlation_model="glm-4.6V",
    )
    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "视频类型：教学\n评分依据：教学画面无需字面对应\n输入图片：老师讲课\n输入文本：知识讲解\n场景合理性：0.8\n内容匹配度：0.7\n置信度：0.8"
                        }
                    }
                ]
            }

    def fake_post(url, headers, json, timeout):
        captured["prompt"] = json["messages"][0]["content"][0]["text"]
        return FakeResponse()

    monkeypatch.setattr(correlation_agent.httpx, "post", fake_post)
    scores = correlation_agent._score_frame_text_batch([(frame, "这是一段教学讲解文本")], settings)

    assert scores[0].confidence == 0.8
    assert "【第一步】视频类型判断" in captured["prompt"]


def test_batch_scoring_failure_is_not_treated_as_medium_relevance(monkeypatch, tmp_path: Path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    frame = VideoFrame(frame_id="f0", timestamp=1.0, path=str(tmp_path / "f0.jpg"))

    monkeypatch.setattr(correlation_agent, "sample_random_correlation_frames", lambda *args, **kwargs: [frame])
    monkeypatch.setattr(
        correlation_agent,
        "_score_frame_text_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("endpoint rejected multimodal content")),
    )

    profile = correlation_agent.assess_video_text_correlation(
        "v1",
        [TranscriptSegment(segment_id="s0", start=0, end=10, text="这是一段足够长的教学台词，用于测试相关性分析失败后的降级行为。")],
        video_path=video,
        output_dir=tmp_path,
        settings=Settings(
            _env_file=None,
            correlation_text_min_chars=5,
            llm_correlation_api_key="key",
            llm_correlation_base_url="https://example.test/v4",
            llm_correlation_model="glm-4.6V",
        ),
    )

    assert profile.status == "failed"
    assert profile.correlation_available is False
    assert profile.relevance_level == "high"
    assert profile.avg_confidence == 1.0
    assert profile.fallback_reason.startswith("batch_scoring_failed:")


def test_unparseable_correlation_response_is_rejected(monkeypatch, tmp_path: Path):
    image = tmp_path / "f0.jpg"
    image.write_bytes(b"fake-image")
    frame = VideoFrame(frame_id="f0", timestamp=1.0, path=str(image))
    settings = Settings(
        _env_file=None,
        llm_correlation_api_key="key",
        llm_correlation_base_url="https://example.test/v4",
        llm_correlation_model="glm-4.6V",
    )

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "无法解析评分"}}]}

    monkeypatch.setattr(correlation_agent.httpx, "post", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(ValueError, match="no parseable confidence scores"):
        correlation_agent._score_frame_text_batch([(frame, "教学台词")], settings)
