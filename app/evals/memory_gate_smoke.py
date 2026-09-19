"""Run synthetic routing examples against the configured real LLM (incurs API usage).

Usage: python -m app.evals.memory_gate_smoke
Does not query or write MemPalace. This is a smoke check, not an accuracy benchmark.
"""
import json
from pathlib import Path
from time import perf_counter

from app.config import get_settings
from app.reasoning.memory_gate import select_memory_route


CASES = [
    ("总结当前视频的核心观点。", [], "current_video_only"),
    ("比较这个视频里先后介绍的两种方法。", [], "current_video_only"),
    ("与我之前看过的其他视频相比，这个方法有什么局限？", [], "search_history"),
    ("当前视频第5分钟的说法，与之前其他视频的观点一致吗？", [], "search_history"),
    ("只依据当前视频说明适用条件，不要引入历史视频。", [], "current_video_only"),
    ("那它们在哪些条件下结论相反？", [
        {"user_question": "比较当前视频和昨天看过的视频", "answer": "两个视频的方法适用范围不同。"}
    ], "search_history"),
]


def main():
    settings = get_settings().model_copy(update={"memory_gate_mode": "auto", "mempalace_provider": "mcp_stdio"})
    rows = []
    for question, history, expected in CASES:
        start = perf_counter()
        result = select_memory_route(question, "训练方法与适用条件", [], history, None, settings)
        row = {"question": question, "expected": expected, **result.model_dump(),
               "seconds": round(perf_counter() - start, 2),
               "passed": result.source == "llm" and result.route == expected}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {"model": settings.memory_gate_model or settings.runtime_llm_model,
              "passed": sum(row["passed"] for row in rows), "total": len(rows), "cases": rows}
    path = Path("data/eval/memory_gate_smoke.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {path}", flush=True)


if __name__ == "__main__":
    main()
