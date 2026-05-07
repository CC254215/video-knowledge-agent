from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _ensure_project_venv() -> None:
    """Run the Web UI with the project venv to avoid global package ABI drift."""
    project_root = Path(__file__).resolve().parent
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return
    current_python = Path(sys.executable).resolve()
    if current_python == venv_python.resolve():
        return
    if os.environ.get("VKA_SKIP_VENV_REEXEC") == "1":
        return
    os.environ["VKA_SKIP_VENV_REEXEC"] = "1"
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


_ensure_project_venv()

from dotenv import load_dotenv

from app.config import get_settings
from app.ui.gradio_app import THEME_CSS, build_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the local Video Knowledge Agent Web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind. Default: 7860")
    parser.add_argument("--share", action="store_true", help="Create a Gradio share link.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    get_settings().log_runtime_config()
    demo = build_app(debug=args.debug)
    try:
        demo.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=True, debug=args.debug, css=THEME_CSS)
    except TypeError:
        demo.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=True, debug=args.debug)


if __name__ == "__main__":
    main()
