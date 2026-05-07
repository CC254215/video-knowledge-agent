from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.runtime.scheduler import get_scheduler

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        settings: Settings | None = None,
        log_dir: Path | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        validate_runtime_endpoint: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.log_dir = log_dir
        self.api_key = api_key or self.settings.openai_api_key
        self.base_url = base_url or self.settings.runtime_llm_base_url
        self.model = model or self.settings.runtime_llm_model
        self.validate_runtime_endpoint = validate_runtime_endpoint
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _write_log(self, name: str, payload: dict[str, Any]) -> None:
        if not self.log_dir:
            return
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
        (self.log_dir / f"{stamp}-{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def generate_text(self, prompt: str) -> str:
        return self.generate_chat([{"role": "user", "content": prompt}])

    def generate_chat(self, messages: list[dict[str, str]], response_format: dict[str, str] | None = None) -> str:
        if not self.api_key or not self.base_url or not self.model:
            raise RuntimeError("Missing LLM config: set OPENAI_API_KEY, LLM_BASE_URL, and LLM_SUMMARY_MODEL.")
        if self.validate_runtime_endpoint:
            self.settings.validate_runtime_config()
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for LLM API calls. Install project dependencies.") from exc
        url = _join_openai_endpoint(self.base_url, "chat/completions")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if response_format:
            payload["response_format"] = response_format
        request_summary = _payload_summary(payload)
        def _request() -> Any:
            result = httpx.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.settings.request_timeout_seconds,
            )
            if result.status_code == 429 or result.status_code >= 500:
                raise RuntimeError(f"retryable_http_status={result.status_code}, body={result.text[:1000]}")
            return result

        try:
            response = get_scheduler(self.settings).call("llm", _request)
            if response.status_code >= 400:
                body = response.text[:4000]
                self._write_log(
                    "chat-error",
                    {
                        "request_summary": request_summary,
                        "status_code": response.status_code,
                        "response_body": body,
                        "url": url,
                        "base_url": self.base_url,
                        "model": self.model,
                    },
                )
                raise RuntimeError(
                    f"LLM request failed: status={response.status_code}, model={self.model}, "
                    f"base_url={self.base_url}, url={url}, "
                    f"request_summary={request_summary}, body={body}"
                )
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            self._write_log("chat", {"request_summary": request_summary, "response": data, "url": url, "model": self.model})
            return text
        except httpx.TimeoutException as exc:
            diagnostic = {
                "request_summary": request_summary,
                "error_type": "timeout",
                "timeout_seconds": self.settings.request_timeout_seconds,
                "url": url,
                "base_url": self.base_url,
                "model": self.model,
            }
            logger.error("LLM request timed out: %s", diagnostic)
            self._write_log("chat-error", diagnostic)
            raise RuntimeError(
                f"LLM request timed out after {self.settings.request_timeout_seconds}s: "
                f"model={self.model}, base_url={self.base_url}, "
                f"url={url}, request_summary={request_summary}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM request failed: %s", exc)
            self._write_log(
                "chat-error",
                {
                    "request_summary": request_summary,
                    "error": str(exc),
                    "url": url,
                    "base_url": self.base_url,
                    "model": self.model,
                },
            )
            raise RuntimeError(f"LLM request failed: {exc}") from exc

    def generate_json(self, prompt: str, schema_hint: str | None = None) -> dict[str, Any]:
        json_prompt = f"{prompt}\nReturn valid JSON only."
        if schema_hint:
            json_prompt += f"\nSchema hint:\n{schema_hint}"
        text = self.generate_text(json_prompt)
        parsed = _parse_json_object(text)
        if parsed is not None:
            return parsed
        repair_prompt = f"Repair this into valid JSON only:\n{text}"
        repaired = self.generate_text(repair_prompt)
        repaired_parsed = _parse_json_object(repaired)
        if repaired_parsed is not None:
            return repaired_parsed
        self._write_log("json-parse-error", {"raw": text, "repaired": repaired})
        raise RuntimeError("LLM returned invalid JSON after one repair attempt.")


def _join_openai_endpoint(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith(endpoint):
        return base
    return f"{base}/{endpoint}"


def _payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages") or []
    return {
        "model": payload.get("model"),
        "messages_count": len(messages),
        "total_message_chars": sum(len(str(item.get("content", ""))) for item in messages if isinstance(item, dict)),
        "has_response_format": "response_format" in payload,
        "temperature": payload.get("temperature"),
    }


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        try:
            value = json.loads(stripped)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None
