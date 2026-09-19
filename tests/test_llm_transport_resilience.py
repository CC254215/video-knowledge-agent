from app.config import Settings
from app.reasoning import llm_client
from app.runtime.scheduler import _is_retryable


class _Response:
    status_code = 200

    def json(self):
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}


def test_zhipu_json_request_disables_thinking_and_constrains_output(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update(json)
        return _Response()

    monkeypatch.setattr("httpx.post", fake_post)
    settings = Settings(
        _env_file=None,
        LLM_API_KEY="test-key",
        LLM_PROVIDER="zhipu",
        LLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4",
        LLM_SUMMARY_MODEL="glm-5.1",
        LLM_THINKING_MODE="disabled",
        LLM_MAX_OUTPUT_TOKENS=2048,
        LLM_MIN_INTERVAL_SECONDS=0,
    )

    result = llm_client.LLMClient(settings).generate_json("Return an object")

    assert result == {"ok": True}
    assert captured["thinking"] == {"type": "disabled"}
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["max_tokens"] == 2048


def test_server_disconnect_is_retryable():
    assert _is_retryable(RuntimeError("Server disconnected without sending a response.")) is True
