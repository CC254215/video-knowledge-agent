from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich import print

from app import pipeline
from app.config import get_settings
from app.memory.mempalace_adapter import MemPalaceMCPAdapter, build_memory_context, create_memory_adapter

app = typer.Typer(help="Local-first Video Knowledge Agent MVP")


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    get_settings().log_runtime_config()


@app.command()
def ingest(
    url: Optional[str] = typer.Option(None, "--url"),
    file: Optional[Path] = typer.Option(None, "--file", exists=True),
    mock: bool = typer.Option(False, "--mock", help="Use built-in mock transcript."),
) -> None:
    _setup_logging()
    try:
        if mock:
            video_id = pipeline.ingest_mock()
        elif url:
            video_id = pipeline.ingest_url(url)
        elif file:
            video_id = pipeline.ingest_local_file(file)
        else:
            raise typer.BadParameter("Provide --mock, --url, or --file.")
        print(f"[green]Ingested video_id={video_id}[/green]")
    except Exception as exc:  # noqa: BLE001
        print(f"[red]Ingest failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def summarize(video_id: str = typer.Option(..., "--video-id")) -> None:
    _setup_logging()
    try:
        report = pipeline.summarize(video_id)
        pipeline.build_storyline(video_id)
        print(report.model_dump_json(indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f"[red]Summarize failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def ask(video_id: str = typer.Option(..., "--video-id"), question: str = typer.Option(..., "--question")) -> None:
    _setup_logging()
    try:
        answer = pipeline.ask(video_id, question)
        print(answer.model_dump_json(indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f"[red]Ask failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def chat(
    video_id: str = typer.Option(..., "--video-id"),
    question: Optional[str] = typer.Option(None, "--question", help="Ask one question and exit."),
) -> None:
    _setup_logging()
    if question:
        try:
            turn = pipeline.chat(video_id, question)
            print(turn.model_dump_json(indent=2))
        except Exception as exc:  # noqa: BLE001
            print(f"[red]Chat failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        return
    print("[green]Video Knowledge Agent chat started. Type 'exit' to quit.[/green]")
    while True:
        question = typer.prompt("You")
        if question.strip().lower() in {"exit", "quit"}:
            break
        try:
            turn = pipeline.chat(video_id, question)
            print(turn.model_dump_json(indent=2))
        except Exception as exc:  # noqa: BLE001
            print(f"[red]Chat failed:[/red] {exc}")


@app.command("refine-frames")
def refine_frames(
    video_id: str = typer.Option(..., "--video-id"),
    start: float = typer.Option(..., "--start"),
    end: float = typer.Option(..., "--end"),
) -> None:
    _setup_logging()
    try:
        segments = pipeline.refine_frames(video_id, start, end)
        print(f"[green]Refined frames for {len(segments)} multimodal segments.[/green]")
    except Exception as exc:  # noqa: BLE001
        print(f"[red]Frame refinement failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("export-obsidian")
def export_obsidian(video_id: str = typer.Option(..., "--video-id")) -> None:
    _setup_logging()
    try:
        paths = pipeline.export_obsidian(video_id)
        print({key: str(value) for key, value in paths.items()})
    except Exception as exc:  # noqa: BLE001
        print(f"[red]Obsidian export failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("rebuild-index")
def rebuild_index(video_id: str = typer.Option(..., "--video-id")) -> None:
    _setup_logging()
    try:
        result = pipeline.rebuild_index(video_id)
        print(result)
    except Exception as exc:  # noqa: BLE001
        print(f"[red]Rebuild index failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def audit(video_id: str = typer.Option(..., "--video-id")) -> None:
    _setup_logging()
    try:
        result = pipeline.audit(video_id)
        print(result)
        if not result.get("ok"):
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[red]Audit failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("quality-gate")
def quality_gate(
    video_id: str = typer.Option(..., "--video-id"),
    start: float = typer.Option(0.0, "--start"),
    end: float = typer.Option(30.0, "--end"),
    question: str = typer.Option("quality gate frame refinement", "--question"),
) -> None:
    """Modified file: app/cli.py. Feature module: quality gate composition.

    Runs refine-frames -> audit -> show-config as one auditable command. Rollback policy: this
    command never deletes artifacts; if refinement fails, audit/config still report where possible.
    """
    _setup_logging()
    result = {"status": "success", "steps": {}, "warnings": [], "errors": []}
    try:
        segments = pipeline.refine_frames(video_id, start, end, question=question)
        result["steps"]["refine_frames"] = {"status": "success", "segments": len(segments)}
    except Exception as exc:  # noqa: BLE001
        result["status"] = "failure"
        result["steps"]["refine_frames"] = {"status": "failure", "error": str(exc).splitlines()[0]}
        result["errors"].append("refine_frames_failed")
    try:
        audit_result = pipeline.audit(video_id)
        result["steps"]["audit"] = audit_result
        if audit_result.get("warnings"):
            result["warnings"].extend(audit_result.get("warnings", []))
        if not audit_result.get("ok"):
            result["status"] = "failure"
            result["errors"].extend(audit_result.get("errors", []))
    except Exception as exc:  # noqa: BLE001
        result["status"] = "failure"
        result["steps"]["audit"] = {"status": "failure", "error": str(exc).splitlines()[0]}
        result["errors"].append("audit_failed")
    result["steps"]["show_config"] = _config_payload()
    if result["warnings"] and result["status"] == "success":
        result["status"] = "warnings"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "failure":
        raise typer.Exit(code=1)


@app.command()
def process(
    url: Optional[str] = typer.Option(None, "--url"),
    file: Optional[Path] = typer.Option(None, "--file", exists=True),
    query: Optional[str] = typer.Option(None, "--query"),
    export_obsidian: bool = typer.Option(False, "--export-obsidian"),
    mock: bool = typer.Option(False, "--mock"),
    force_refresh: bool = typer.Option(False, "--force-refresh"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    force_stage: Optional[str] = typer.Option(None, "--force-stage", help="Re-run one stage: multimodal/index/summary/storyline/export_obsidian."),
    no_ocr: bool = typer.Option(False, "--no-ocr"),
    strict_llm: bool = typer.Option(False, "--strict-llm"),
) -> None:
    _setup_logging()
    try:
        settings = get_settings()
        if force_refresh:
            settings.force_refresh = True
        settings.pipeline_resume = resume
        if force_stage:
            settings.pipeline_force_stage = force_stage
        if no_ocr:
            settings.enable_ocr = False
            settings.ocr_provider = "none"
            settings.ocr_engine = "none"
        if strict_llm:
            settings.strict_llm = True
            settings.strict_runtime = True
        result = pipeline.process(url=url, file_path=file, query=query, export_to_obsidian=export_obsidian, mock=mock, settings=settings)
        _print_process_report(result)
        if result.get("obsidian"):
            print({key: str(value) for key, value in result["obsidian"].items()})  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        print(f"[red]Process failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def _print_process_report(result: dict[str, object]) -> None:
    summary = result.get("summary")
    storyline = result.get("storyline")
    index_stats = result.get("index_stats") or {}
    print("[green]Pipeline Status: success[/green]")
    print(f"video_id={result['video_id']}")
    if result.get("llm_timeout_seconds"):
        print(f"Dynamic LLM timeout: {float(result['llm_timeout_seconds']) / 60:.1f} minutes")
    print(f"LLM Summary: {getattr(summary, 'generation_status', 'unknown')}")
    print("Evidence Mode: speech + frame_caption, OCR disabled")
    if isinstance(index_stats, dict):
        print(f"ChromaDB indexed: speech={index_stats.get('speech_documents', 0)}, frame_caption={index_stats.get('frame_caption_documents', 0)}")
    memory_stats = result.get("memory")
    if isinstance(memory_stats, dict):
        print(f"MemPalace memory: status={memory_stats.get('status')} model_summary={memory_stats.get('model_summary', 0)} raw_evidence={memory_stats.get('raw_evidence', 0)}")
    timings = result.get("stage_timings")
    if isinstance(timings, dict) and timings:
        print("\n[bold]Stage Timings[/bold]")
        for name, seconds in sorted(timings.items(), key=lambda item: str(item[0])):
            print(f"- {name}: {float(seconds):.2f}s")
    if result.get("processing_state_path"):
        print(f"processing_state={result['processing_state_path']}")
    if summary:
        print("\n[bold]30 秒速览[/bold]")
        for item in getattr(summary, "quick_overview", []):
            print(f"- {item}")
        print("\n[bold]Structured Outline[/bold]")
        for item in getattr(summary, "structured_outline", [])[:8]:
            print(f"- {item.timestamp} {item.topic} ({', '.join(item.segment_ids)})")
    if storyline:
        print("\n[bold]Query-Guided Storyline[/bold]")
        for node in getattr(storyline, "nodes", [])[:8]:
            title = getattr(node, "title", None) or node.topic
            print(f"- {node.node_id} {node.time_start:.0f}s-{node.time_end:.0f}s {title}: {getattr(node, 'summary', None) or node.claim} status={node.status.value}")


@app.command("show-config")
def show_config() -> None:
    print(_config_payload())


@app.command("mempalace-status")
def mempalace_status() -> None:
    _setup_logging()
    try:
        adapter = create_memory_adapter()
        if not isinstance(adapter, MemPalaceMCPAdapter):
            print({"status": "disabled", "provider": get_settings().mempalace_provider})
            return
        status = adapter.status()
        tools = adapter.list_tools()
        print({"status": status, "tool_count": len(tools), "tools": tools})
        adapter.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[red]MemPalace status failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("mempalace-search")
def mempalace_search(query: str = typer.Argument(...), limit: int = typer.Option(5, "--limit")) -> None:
    _setup_logging()
    try:
        context = build_memory_context(query, limit=limit)
        print(json.dumps(context, ensure_ascii=False, indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f"[red]MemPalace search failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def _config_payload() -> dict[str, object]:
    settings = get_settings()
    return {
        "data_dir": str(settings.data_dir),
        "obsidian_vault_path": str(settings.obsidian_vault_path) if settings.obsidian_vault_path else None,
        "has_llm_config": settings.has_llm_config,
        "has_embedding_config": settings.has_embedding_config,
        "has_vision_config": settings.has_vision_config,
        "asr_provider": settings.asr_provider,
        "asr_model": settings.asr_model,
        "mempalace_provider": settings.mempalace_provider,
    }


if __name__ == "__main__":
    app()
