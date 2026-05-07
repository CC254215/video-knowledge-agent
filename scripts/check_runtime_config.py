from __future__ import annotations

import json
import logging

from app.config import get_settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    settings.validate_runtime_config()
    print(json.dumps(settings.runtime_config_summary(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
