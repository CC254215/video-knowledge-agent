from __future__ import annotations

import logging
import json
import hashlib
from abc import ABC, abstractmethod
from typing import Any

from app.config import Settings, get_settings
from app.memory.mcp_stdio_client import MCPStdioClient, MCPStdioError, build_mempalace_args

logger = logging.getLogger(__name__)


class MemoryAdapter(ABC):
    @abstractmethod
    def save_conversation_turn(self, payload: dict[str, Any]) -> None: ...

    @abstractmethod
    def save_raw_evidence(self, payload: dict[str, Any]) -> None: ...

    @abstractmethod
    def save_model_summary(self, payload: dict[str, Any]) -> None: ...

    @abstractmethod
    def save_user_verified_insight(self, payload: dict[str, Any]) -> None: ...

    @abstractmethod
    def search_related_memories(self, query: str, limit: int = 5) -> list[dict[str, Any]]: ...


class NoOpMemoryAdapter(MemoryAdapter):
    def save_conversation_turn(self, payload: dict[str, Any]) -> None:
        logger.info("NoOpMemoryAdapter skipped conversation save: %s", payload.get("video_id"))

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

    def save_conversation_turn(self, payload: dict[str, Any]) -> None:
        raise RuntimeError(
            "Verbatim conversation storage requires the native MemPalace MCP provider; "
            "set MEMPALACE_PROVIDER=mcp_stdio"
        )

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
            raise RuntimeError("MemPalace endpoint missing")
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx is not installed")
        try:
            response = httpx.post(
                f"{self.endpoint}{path}",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"MemPalace write failed: {path}") from exc

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class MemPalaceMCPAdapter(MemoryAdapter):
    """MemPalace adapter backed by its native MCP stdio server."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        environment = {"MEMPALACE_EMBEDDING_MODEL": self.settings.mempalace_embedding_model,
                       "PYTHONUTF8": "1"}
        if self.settings.mempalace_embedding_model == "openai-compat":
            if not self.settings.has_embedding_config:
                raise ValueError("MemPalace requires the configured project embedding API")
            base = self.settings.runtime_embedding_base_url.rstrip("/")
            environment.update({"MEMPALACE_EMBEDDING_API_URL": base if base.endswith("/embeddings") else base + "/embeddings",
                                "MEMPALACE_EMBEDDING_API_MODEL": self.settings.embedding_model,
                                "MEMPALACE_EMBEDDING_API_KEY": self.settings.openai_api_key})
        self.client = MCPStdioClient(
            command=self.settings.mempalace_command,
            args=build_mempalace_args(self.settings.mempalace_args, self.settings.mempalace_palace_path),
            timeout_seconds=self.settings.mempalace_timeout_seconds,
            env=environment,
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
            result = self.client.call_tool(name, arguments or {})
            if result.get("isError"):
                raise MCPStdioError(str(result.get("content")))
            for block in result.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    try:
                        decoded = json.loads(block.get("text", ""))
                    except (ValueError, TypeError):
                        continue
                    if isinstance(decoded, dict):
                        if decoded.get("success") is False or decoded.get("error"):
                            raise MCPStdioError(str(decoded))
                        return decoded
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("MemPalace MCP tool call failed: tool=%s error=%s", name, exc)
            return {"error": str(exc), "tool": name}

    def save_raw_evidence(self, payload: dict[str, Any]) -> None:
        self._add_drawer("raw_evidence", payload)

    def save_conversation_turn(self, payload: dict[str, Any]) -> None:
        """File an unmodified user/assistant exchange as a native MemPalace drawer.

        The conversation text deliberately is not converted to a generated
        summary.  Metadata used by this application stays outside the drawer
        body so MemPalace indexes the actual exchange.
        """
        content = str(payload.get("content") or "").strip()
        if not content:
            raise ValueError("conversation drawer content cannot be empty")
        result = self.call_tool("mempalace_add_drawer", {
            "content": content,
            "wing": "video_knowledge_agent",
            "room": str(payload.get("room") or "conversations"),
            "source_file": str(payload.get("source_file") or "conversation.jsonl"),
            "added_by": "video_knowledge_agent",
        })
        if result.get("error"):
            raise MCPStdioError(str(result["error"]))

    def save_model_summary(self, payload: dict[str, Any]) -> None:
        self._add_drawer("model_summary", payload)

    def save_user_verified_insight(self, payload: dict[str, Any]) -> None:
        self._add_drawer("user_verified", payload)

    def save_derived_insight(self, payload: dict[str, Any]) -> None:
        self._add_drawer("derived_insight", payload)

    def search_related_memories(self, query: str, limit: int = 5, room: str | None = None) -> list[dict[str, Any]]:
        arguments = {"query": query, "limit": max(1, int(limit)), "wing": "video_knowledge_agent"}
        if room:
            arguments["room"] = room
        result = self.call_tool("mempalace_search", arguments)
        if result.get("error"):
            raise MCPStdioError(str(result["error"]))
        return _normalize_search_result(result)

    def write_diary(self, content: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call_tool("mempalace_diary_write", {"entry": content, "agent_name": "video_knowledge_agent", "topic": (metadata or {}).get("topic", "general")})

    def close(self) -> None:
        self.client.close()

    def _add_drawer(self, memory_type: str, payload: dict[str, Any]) -> None:
        content = _drawer_content(memory_type, payload)
        result = self.call_tool("mempalace_add_drawer", {
            "content": content,
            "wing": "video_knowledge_agent",
            "room": payload.get("primary_topic_id") or memory_type,
            "source_file": payload.get("source_file") or "vka_" + hashlib.sha256(content.encode()).hexdigest() + ".json",
            "added_by": "video_knowledge_agent",
        })
        if result.get("error"):
            raise MCPStdioError(str(result["error"]))


def build_memory_context(
    query: str,
    adapter: MemoryAdapter | None = None,
    limit: int = 5,
    token_budget_chars: int = 3000,
) -> dict[str, Any]:
    """Build structured long-term memory context without mixing it with video evidence."""
    owned = adapter is None
    adapter = adapter or create_memory_adapter()
    try:
        rows = adapter.search_related_memories(query, limit=limit)
    except Exception as exc:
        logger.warning("Long-term memory unavailable: %s", exc)
        rows = []
    finally:
        close = getattr(adapter, "close", None)
        if owned and callable(close):
            close()
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
            continue
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
    body = {"memory_type": memory_type, **payload}
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
