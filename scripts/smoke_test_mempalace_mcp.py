from __future__ import annotations

import json
import sys

from app.config import get_settings
from app.memory.mempalace_adapter import MemPalaceMCPAdapter


def main() -> int:
    settings = get_settings()
    adapter = MemPalaceMCPAdapter(settings)
    try:
        status = adapter.status()
        tools = adapter.list_tools()
        search = adapter.search_related_memories("video knowledge agent", limit=3)
        print(
            json.dumps(
                {
                    "provider": "mcp_stdio",
                    "command": settings.mempalace_command,
                    "args": settings.mempalace_args,
                    "palace_path": str(settings.mempalace_palace_path) if settings.mempalace_palace_path else None,
                    "status": status,
                    "tool_count": len(tools),
                    "has_search": "mempalace_search" in tools,
                    "search_preview": search[:3],
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 0 if tools else 1
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
