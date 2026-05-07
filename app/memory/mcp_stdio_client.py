from __future__ import annotations

import json
import logging
import queue
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MCPStdioError(RuntimeError):
    pass


@dataclass
class MCPTool:
    name: str
    description: str = ""
    input_schema: dict[str, Any] | None = None


class MCPStdioClient:
    """Minimal MCP JSON-RPC client over stdio.

    MCP stdio messages are JSON-RPC payloads framed with a Content-Length
    header. This client keeps one server process alive for low-latency tool
    calls.
    """

    def __init__(
        self,
        command: str = "python",
        args: list[str] | None = None,
        timeout_seconds: float = 30.0,
        cwd: str | Path | None = None,
    ) -> None:
        self.command = command
        self.args = args or []
        self.timeout_seconds = timeout_seconds
        self.cwd = str(cwd) if cwd else None
        self._process: subprocess.Popen[bytes] | None = None
        self._next_id = 1
        self._pending: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._notifications: queue.Queue[dict[str, Any]] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._initialized = False
        self.stderr_lines: list[str] = []

    def start(self) -> None:
        if self._process and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            [self.command, *self.args],
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        self._reader_thread = threading.Thread(target=self._read_loop, name="mcp-stdio-reader", daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._stderr_loop, name="mcp-stdio-stderr", daemon=True)
        self._stderr_thread.start()

    def initialize(self) -> dict[str, Any]:
        self.start()
        if self._initialized:
            return {"initialized": True}
        result = self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "video-knowledge-agent", "version": "0.1.0"},
            },
        )
        self.notify("notifications/initialized", {})
        self._initialized = True
        return result

    def list_tools(self) -> list[MCPTool]:
        self.initialize()
        result = self.request("tools/list", {})
        tools = result.get("tools") if isinstance(result, dict) else []
        output: list[MCPTool] = []
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            output.append(
                MCPTool(
                    name=str(tool.get("name") or ""),
                    description=str(tool.get("description") or ""),
                    input_schema=tool.get("inputSchema") if isinstance(tool.get("inputSchema"), dict) else None,
                )
            )
        return output

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        self.initialize()
        return self.request("tools/call", {"name": name, "arguments": arguments or {}})

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.start()
        request_id = self._reserve_id()
        response_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._pending[request_id] = response_queue
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        try:
            response = response_queue.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            self._pending.pop(request_id, None)
            raise MCPStdioError(f"MCP request timed out: method={method}, timeout={self.timeout_seconds}s") from exc
        if "error" in response:
            raise MCPStdioError(f"MCP request failed: method={method}, error={response['error']}")
        result = response.get("result")
        return result if isinstance(result, dict) else {"result": result}

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self.start()
        self._write({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def close(self) -> None:
        process = self._process
        if not process:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
        finally:
            self._process = None
            self._initialized = False

    def _reserve_id(self) -> int:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            return request_id

    def _write(self, payload: dict[str, Any]) -> None:
        process = self._process
        if not process or not process.stdin or process.poll() is not None:
            raise MCPStdioError("MCP server is not running.")
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        frame = f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii") + raw
        try:
            process.stdin.write(frame)
            process.stdin.flush()
        except BrokenPipeError as exc:
            raise MCPStdioError("MCP server stdin pipe is closed.") from exc

    def _read_loop(self) -> None:
        assert self._process and self._process.stdout
        stream = self._process.stdout
        while True:
            try:
                payload = _read_mcp_message(stream)
            except Exception as exc:  # noqa: BLE001
                if self._process and self._process.poll() is None:
                    logger.debug("MCP read loop stopped: %s", exc)
                return
            if payload is None:
                return
            request_id = payload.get("id")
            if isinstance(request_id, int) and request_id in self._pending:
                self._pending.pop(request_id).put(payload)
            else:
                self._notifications.put(payload)

    def _stderr_loop(self) -> None:
        assert self._process and self._process.stderr
        while True:
            line = self._process.stderr.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                self.stderr_lines.append(text)
                self.stderr_lines = self.stderr_lines[-100:]
                logger.debug("MCP stderr: %s", text)

    def __enter__(self) -> "MCPStdioClient":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()


def build_mempalace_args(raw_args: str, palace_path: str | Path | None = None) -> list[str]:
    args = shlex.split(raw_args, posix=False)
    if palace_path:
        args.extend(["--palace", str(palace_path)])
    return args


def _read_mcp_message(stream) -> dict[str, Any] | None:  # type: ignore[no-untyped-def]
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        if line in {b"\r\n", b"\n"}:
            break
        key, _, value = line.decode("ascii", errors="replace").partition(":")
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length") or 0)
    if length <= 0:
        raise MCPStdioError("Invalid MCP frame: missing Content-Length.")
    body = stream.read(length)
    if len(body) != length:
        raise MCPStdioError("Invalid MCP frame: truncated body.")
    return json.loads(body.decode("utf-8"))


def wait_for_process_exit(client: MCPStdioClient, timeout_seconds: float = 1.0) -> int | None:
    process = client._process
    if not process:
        return None
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        code = process.poll()
        if code is not None:
            return code
        time.sleep(0.05)
    return process.poll()
