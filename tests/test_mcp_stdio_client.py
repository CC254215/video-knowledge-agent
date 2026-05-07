from __future__ import annotations

import json
from pathlib import Path

from app.memory.mcp_stdio_client import MCPStdioClient, build_mempalace_args


MOCK_SERVER = r'''
import json
import sys

def read_msg():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        key, _, value = line.decode("ascii").partition(":")
        headers[key.lower().strip()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body.decode("utf-8"))

def write_msg(payload):
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii") + raw)
    sys.stdout.buffer.flush()

while True:
    msg = read_msg()
    if msg is None:
        break
    if "id" not in msg:
        continue
    method = msg.get("method")
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}}
    elif method == "tools/list":
        result = {"tools": [{"name": "mempalace_status"}, {"name": "mempalace_search"}]}
    elif method == "tools/call":
        name = msg.get("params", {}).get("name")
        if name == "mempalace_status":
            result = {"content": [{"type": "text", "text": "ok"}]}
        else:
            result = {"results": [{"content": "memory result", "memory_type": "model_summary"}]}
    else:
        result = {}
    write_msg({"jsonrpc": "2.0", "id": msg["id"], "result": result})
'''


def test_mcp_stdio_client_lists_and_calls_tools(tmp_path: Path) -> None:
    server = tmp_path / "mock_mcp_server.py"
    server.write_text(MOCK_SERVER, encoding="utf-8")
    client = MCPStdioClient(command="python", args=[str(server)], timeout_seconds=5)
    try:
        tools = client.list_tools()
        assert "mempalace_status" in {tool.name for tool in tools}
        result = client.call_tool("mempalace_status")
        assert result["content"][0]["text"] == "ok"
    finally:
        client.close()


def test_build_mempalace_args_adds_palace_path() -> None:
    args = build_mempalace_args("-m mempalace.mcp_server", "D:/palace")
    assert args[:2] == ["-m", "mempalace.mcp_server"]
    assert args[-2:] == ["--palace", "D:/palace"]
