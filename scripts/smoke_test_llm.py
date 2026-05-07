from __future__ import annotations

import logging

import httpx

from app.config import get_settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    settings.log_runtime_config()
    url = f"{settings.runtime_llm_base_url}/chat/completions"
    payload = {
        "model": settings.runtime_llm_model,
        "messages": [
            {"role": "system", "content": "You are a strict JSON generator."},
            {"role": "user", "content": 'Return {"ok": true} only.'},
        ],
        "temperature": 0,
    }
    print(f"provider={settings.llm_provider}")
    print(f"base_url={settings.runtime_llm_base_url}")
    print(f"model={settings.runtime_llm_model}")
    try:
        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json=payload,
            timeout=settings.request_timeout_seconds,
        )
        print(f"HTTP status={response.status_code}")
        print("response text:")
        print(response.text)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print("error response body:")
        print(exc.response.text)
        raise


if __name__ == "__main__":
    main()
