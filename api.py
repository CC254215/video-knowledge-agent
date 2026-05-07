from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from app.config import get_settings
from app.api.server import create_app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Video Knowledge Agent API server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind. Default: 8000")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    get_settings().log_runtime_config()

    import uvicorn

    uvicorn.run("api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()

