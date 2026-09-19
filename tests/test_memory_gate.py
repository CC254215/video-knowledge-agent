import json
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.reasoning import conversation_agent as conversation
from app.reasoning import memory_gate


def settings(**overrides):
    return Settings(_env_file=None, openai_api_key="test-key", mempalace_provider="mcp_stdio",
                    **overrides)


@pytest.mark.parametrize("route", ["current_video_only", "search_history"])
def test_structured_semantic_selection(monkeypatch, route):
    captured = {}

    def chat(self, messages, response_format):
        captured.update(settings=self.settings, prompt=messages[0]["content"], model=self.model)
        return json.dumps({"route": route, "reason": "依据问题的来源范围", "query": "其他视频的适用条件"})

    monkeypatch.setattr(memory_gate.LLMClient, "generate_chat", chat)
    result = memory_gate.select_memory_route("有哪些限制？", "实验", [], [
        {"user_question": f"history-{i}", "answer": "x" * 1000} for i in range(9)
    ], None, settings(memory_gate_model="route-model"))
    assert result.route == route and result.source == "llm"
    assert captured["model"] == "route-model"
    assert captured["settings"].api_max_retries == 0
    assert captured["settings"].llm_max_output_tokens == 512
    assert "history-2" not in captured["prompt"] and "history-3" in captured["prompt"]
    assert "x" * 601 not in captured["prompt"]


@pytest.mark.parametrize("response", ["not json", "[]", '{"route":"unknown","reason":"x"}',
    '{"route":"search_history","reason":" "}',
    '{"route":"search_history","reason":"x","source":"config"}'])
def test_invalid_model_output_skips_memory(monkeypatch, response):
    monkeypatch.setattr(memory_gate.LLMClient, "generate_chat", lambda *a, **k: response)
    result = memory_gate.select_memory_route("比较历史", "", [], [], None, settings())
    assert result.route == "current_video_only" and result.source == "fallback"


def test_timeout_skips_memory_without_exposing_exception(monkeypatch):
    def fail(*args, **kwargs):
        raise TimeoutError("private details")
    monkeypatch.setattr(memory_gate.LLMClient, "generate_chat", fail)
    result = memory_gate.select_memory_route("比较历史", "", [], [], None, settings())
    assert result.reason == "memory_gate_failed:TimeoutError"


@pytest.mark.parametrize("mode,provider,expected", [
    ("off", "mcp_stdio", "current_video_only"),
    ("always", "mcp_stdio", "search_history"),
    ("always", "noop", "current_video_only"),
])
def test_config_routes_do_not_call_model(monkeypatch, mode, provider, expected):
    monkeypatch.setattr(memory_gate.LLMClient, "generate_chat", lambda *a, **k: pytest.fail("unexpected LLM call"))
    config = settings(memory_gate_mode=mode).model_copy(update={"mempalace_provider": provider})
    assert memory_gate.select_memory_route("问题", "", [], [], None, config).route == expected


def test_blank_search_query_uses_question(monkeypatch):
    monkeypatch.setattr(memory_gate.LLMClient, "generate_chat", lambda *a, **k:
                        '{"route":"search_history","reason":"需要历史","query":" "}')
    assert memory_gate.select_memory_route("比较历史", "", [], [], None, settings()).query == "比较历史"


def test_gate_shares_scheduler_but_does_not_retry(monkeypatch):
    from app.reasoning import llm_client
    from app.runtime.scheduler import ApiScheduler

    config = settings(llm_min_interval_seconds=0, api_max_retries=3)
    seen, requests = [], []
    scheduler = ApiScheduler(config)

    def get_scheduler(original):
        seen.append(original)
        return scheduler

    def unavailable(*args, **kwargs):
        requests.append(kwargs)
        return SimpleNamespace(status_code=429, text="rate limited")

    monkeypatch.setattr(llm_client, "get_scheduler", get_scheduler)
    monkeypatch.setattr("httpx.post", unavailable)
    result = memory_gate.select_memory_route("比较历史", "", [], [], None, config)
    assert len(seen) == 1 and seen[0] is config
    assert len(requests) == 1
    assert requests[0]["timeout"] == config.memory_gate_timeout_seconds
    assert result.source == "fallback"


@pytest.mark.parametrize("route", ["current_video_only", "search_history"])
def test_conversation_retrieval_and_reanswer_reuse(tmp_path, monkeypatch, route):
    selections, searches, events = [], [], []
    agent = conversation.ConversationAgent("v", tmp_path, settings=settings(), store=object())
    monkeypatch.setattr(agent, "_load_metadata", lambda: SimpleNamespace(title="视频"))
    monkeypatch.setattr(agent, "_load_history", lambda: [])
    agent._active_trace = SimpleNamespace(record=lambda name, data: events.append((name, data)))

    def select(*args):
        selections.append(args)
        return memory_gate.MemoryRoute(route=route, reason="语义判断", query="独立历史查询")

    def retrieve(query, *args, **kwargs):
        searches.append(query)
        return {"long_term_memory": [], "status": "ok"}

    monkeypatch.setattr(conversation, "select_memory_route", select)
    monkeypatch.setattr(conversation, "historical_context", retrieve)
    monkeypatch.setattr(conversation.LLMClient, "generate_json", lambda *a, **k: {"answer": "当前视频回答"})
    for _ in range(2):
        assert agent._generate_llm_answer("问题", "explanation", [], False)["answer"] == "当前视频回答"
    assert len(selections) == 1
    assert searches == (["独立历史查询"] * 2 if route == "search_history" else [])
    assert [data["reused_in_turn"] for name, data in events if name == "memory_gate"] == [False, True]

    # The public entry point must clear the decision even when a new question has no segments.
    monkeypatch.setattr(agent, "_load_segments", lambda: [])
    agent.answer("新问题", persist=False)
    assert agent._memory_route is None
