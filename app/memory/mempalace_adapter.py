from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from app.config import Settings, get_settings
from app.memory.mcp_stdio_client import MCPStdioClient, MCPStdioError, build_mempalace_args

logger = logging.getLogger(__name__)


class MemoryAdapter(ABC):
    @abstractmethod
    def save_raw_evidence(self, payload: dict[str, Any]) -> None: ...

    @abstractmethod
    def save_model_summary(self, payload: dict[str, Any]) -> None: ...

    @abstractmethod
    def save_user_verified_insight(self, payload: dict[str, Any]) -> None: ...

    @abstractmethod
    def search_related_memories(self, query: str, limit: int = 5) -> list[dict[str, Any]]: ...


class NoOpMemoryAdapter(MemoryAdapter):
    def save_raw_evidence(self, payload: dict[str, Any]) -> None:
        logger.info("NoOpMemoryAdapter skipped raw_evidence save: %s", payload.get("video_id"))

    def save_model_summary(self, payload: dict[str, Any]) -> None:
        logger.info("NoOpMemoryAdapter skipped model_summary save: %s", payload.get("video_id"))

    def save_user_verified_insight(self, payload: dict[str, Any]) -> None:
        logger.info("NoOpMemoryAdapter skipped user_verified save: %s", payload.get("video_id"))

    def save_derived_insight(self, payload: dict[str, Any]) -> None:
        logger.info("NoOpMemoryAdapter skipped derived_insight save: %s", payload.get("video_id"))

    def search_related_memories(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        logger.info("NoOpMemoryAdapter skipped memory search: %s", query)
        return []


class MemPalaceAdapter(NoOpMemoryAdapter):
    """HTTP adapter for MemPalace-compatible memory APIs."""

    def __init__(self, endpoint: str | None = None, api_key: str | None = None, timeout_seconds: float = 15.0) -> None:
        self.endpoint = (endpoint or "").rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def save_raw_evidence(self, payload: dict[str, Any]) -> None:
        self._post("/memories/raw-evidence", payload)

    def save_model_summary(self, payload: dict[str, Any]) -> None:
        self._post("/memories/model-summary", payload)

    def save_user_verified_insight(self, payload: dict[str, Any]) -> None:
        self._post("/memories/user-verified-insight", payload)

    def save_derived_insight(self, payload: dict[str, Any]) -> None:
        self._post("/memories/derived-insight", payload)

    def search_related_memories(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if not self.endpoint:
            logger.warning("MemPalace endpoint missing; skip memory search.")
            return []
        try:
            import httpx
        except ImportError:
            logger.warning("httpx is not installed; skip MemPalace memory search.")
            return []
        try:
            response = httpx.get(
                f"{self.endpoint}/memories/search",
                params={"query": query, "limit": max(1, int(limit))},
                headers=self._headers(),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                rows = data.get("items") or data.get("results") or []
                return rows if isinstance(rows, list) else []
            return data if isinstance(data, list) else []
        except Exception as exc:  # noqa: BLE001
            logger.warning("MemPalace memory search failed: %s", exc)
            return []

    def _post(self, path: str, payload: dict[str, Any]) -> None:
        if not self.endpoint:
            logger.warning("MemPalace endpoint missing; skip %s", path)
            return
        try:
            import httpx
        except ImportError:
            logger.warning("httpx is not installed; skip MemPalace call %s", path)
            return
        try:
            response = httpx.post(
                f"{self.endpoint}{path}",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MemPalace call failed (%s): %s", path, exc)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class MemPalaceMCPAdapter(MemoryAdapter):
    """MemPalace adapter backed by its native MCP stdio server."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = MCPStdioClient(
            command=self.settings.mempalace_command,
            args=build_mempalace_args(self.settings.mempalace_args, self.settings.mempalace_palace_path),
            timeout_seconds=self.settings.mempalace_timeout_seconds,
        )
        self._available_tools: set[str] | None = None

    def status(self) -> dict[str, Any]:
        return self.call_tool("mempalace_status", {})

    def list_tools(self) -> list[str]:
        tools = self.client.list_tools()
        self._available_tools = {tool.name for tool in tools}
        return sorted(self._available_tools)

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            if self._available_tools is None:
                self.list_tools()
            if self._available_tools and name not in self._available_tools:
                raise MCPStdioError(f"MemPalace MCP tool is unavailable: {name}")
            return self.client.call_tool(name, arguments or {})
        except Exception as exc:  # noqa: BLE001
            logger.warning("MemPalace MCP tool call failed: tool=%s error=%s", name, exc)
            return {"error": str(exc), "tool": name}

    def save_raw_evidence(self, payload: dict[str, Any]) -> None:
        self._add_drawer("raw_evidence", payload)

    def save_model_summary(self, payload: dict[str, Any]) -> None:
        self._add_drawer("model_summary", payload)

    def save_user_verified_insight(self, payload: dict[str, Any]) -> None:
        self._add_drawer("user_verified", payload)

    def save_derived_insight(self, payload: dict[str, Any]) -> None:
        self._add_drawer("derived_insight", payload)

    def search_related_memories(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        result = self.call_tool("mempalace_search", {"query": query, "limit": max(1, int(limit))})
        if result.get("error"):
            return []
        return _normalize_search_result(result)

    def write_diary(self, content: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call_tool("mempalace_diary_write", {"content": content, "metadata": metadata or {}})

    def close(self) -> None:
        self.client.close()

    def _add_drawer(self, memory_type: str, payload: dict[str, Any]) -> None:
        content = _drawer_content(memory_type, payload)
        metadata = {
            "source": "video_knowledge_agent",
            "memory_type": memory_type,
            "video_id": payload.get("video_id"),
            "segment_id": payload.get("segment_id"),
            "evidence_type": payload.get("evidence_type"),
            "timestamp": payload.get("timestamp"),
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}
        result = self.call_tool("mempalace_add_drawer", {"content": content, "metadata": metadata})
        if result.get("error"):
            logger.warning("MemPalace MCP add_drawer skipped: %s", result["error"])


def build_memory_context(
    query: str,
    adapter: MemoryAdapter | None = None,
    limit: int = 5,
    token_budget_chars: int = 3000,
) -> dict[str, Any]:
    """Build structured long-term memory context without mixing it with video evidence."""
    adapter = adapter or create_memory_adapter()
    rows = adapter.search_related_memories(query, limit=limit)
    selected: list[dict[str, Any]] = []
    used = 0
    seen = set()
    for row in rows:
        content = str(row.get("content") or row.get("text") or row.get("memory") or "")
        if not content.strip():
            continue
        fingerprint = content[:200]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        if used + len(content) > token_budget_chars:
            break
        used += len(content)
        selected.append(
            {
                "source": "mempalace",
                "memory_type": row.get("memory_type") or row.get("type") or "unknown",
                "content": content,
                "confidence": "memory",
                "usage_rule": "May guide background context or user preference; cannot support claims about the current video without video evidence.",
                "raw": row,
            }
        )
    return {
        "long_term_memory": selected,
        "answer_policy": {
            "video_claims_require_video_evidence": True,
            "memory_must_be_labeled": True,
            "prefer_current_video_when_conflicting": True,
        },
    }


def _drawer_content(memory_type: str, payload: dict[str, Any]) -> str:
    title = payload.get("title") or payload.get("video_title") or payload.get("video_id") or memory_type
    body = payload.get("content") or payload.get("text") or payload.get("summary") or payload.get("answer") or payload
    if isinstance(body, dict):
        body_text = json_safe(body)
    else:
        body_text = str(body)
    return f"[{memory_type}] {title}\n\n{body_text}"


def _normalize_search_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: Any = result.get("items") or result.get("results") or result.get("drawers")
    if rows is None and isinstance(result.get("content"), list):
        text = "\n".join(str(item.get("text") or "") for item in result["content"] if isinstance(item, dict))
        return [{"content": text, "memory_type": "search_result"}] if text.strip() else []
    if isinstance(rows, list):
        return [row if isinstance(row, dict) else {"content": str(row)} for row in rows]
    if isinstance(rows, dict):
        return [rows]
    text = result.get("result")
    return [{"content": str(text), "memory_type": "search_result"}] if text else []


def json_safe(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def create_memory_adapter(settings: Settings | None = None) -> MemoryAdapter:
    settings = settings or get_settings()
    if settings.mempalace_provider.lower() in {"mcp", "mcp_stdio", "stdio"}:
        adapter = MemPalaceMCPAdapter(settings)
        if settings.mempalace_auto_status:
            adapter.status()
        return adapter
    if settings.mempalace_provider.lower() == "mempalace":
        return MemPalaceAdapter(
            endpoint=settings.mempalace_endpoint,
            api_key=settings.mempalace_api_key,
            timeout_seconds=min(max(float(settings.request_timeout_seconds), 1.0), 60.0),
        )
    return NoOpMemoryAdapter()
