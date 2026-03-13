"""MedRAG Agent CLI"""

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from medrag.agent import MedRAGAgent

app = typer.Typer(
    name="medrag",
    help="의료 도메인 특화 RAG 기반 Q&A Agent",
    add_completion=False,
)
console = Console()


def _get_agent() -> MedRAGAgent:
    return MedRAGAgent()


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="ingestion할 파일 또는 디렉토리 경로"),
):
    """문서를 벡터 DB에 인덱싱합니다."""
    if not path.exists():
        console.print(f"[red]오류: 경로를 찾을 수 없습니다 — {path}[/red]")
        raise typer.Exit(1)

    agent = _get_agent()
    console.print(f"[cyan]Ingestion 시작: {path}[/cyan]")

    with console.status("[bold green]문서 처리 중..."):
        result = agent.ingest(path)

    table = Table(title="Ingestion 결과", show_header=True)
    table.add_column("항목", style="cyan")
    table.add_column("수량", justify="right", style="green")
    table.add_row("로드된 문서", str(result["documents"]))
    table.add_row("생성된 청크", str(result["chunks"]))
    table.add_row("새로 추가된 청크", str(result["added"]))
    table.add_row("총 인덱싱 청크", str(agent.status()["indexed_chunks"]))
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="질문을 입력하세요"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="검색할 청크 수"),
    show_sources: bool = typer.Option(True, "--sources/--no-sources", help="출처 표시 여부"),
    json_output: bool = typer.Option(False, "--json", help="JSON 형식으로 출력"),
):
    """단일 질문에 대한 답변을 생성합니다."""
    agent = _get_agent()

    if agent.status()["indexed_chunks"] == 0:
        console.print("[yellow]경고: 인덱싱된 문서가 없습니다. 먼저 `medrag ingest <경로>`를 실행하세요.[/yellow]")
        raise typer.Exit(1)

    with console.status("[bold green]문서 검색 및 답변 생성 중..."):
        result = agent.query(question, top_k=top_k)

    if json_output:
        print(json.dumps({
            "answer": result.answer,
            "confidence": result.confidence,
            "sources": result.sources,
        }, ensure_ascii=False, indent=2))
        return

    confidence_color = {"high": "green", "medium": "yellow", "low": "red"}.get(
        result.confidence, "white"
    )
    console.print(
        Panel(
            Markdown(result.answer),
            title=f"[bold]답변[/bold] | 신뢰도: [{confidence_color}]{result.confidence}[/{confidence_color}]",
            border_style="blue",
        )
    )

    if show_sources and result.sources:
        table = Table(title="참조 문서", show_header=True, header_style="bold magenta")
        table.add_column("문서", style="cyan")
        table.add_column("페이지", justify="center")
        table.add_column("관련도", justify="right", style="green")
        for src in result.sources:
            table.add_row(
                src["document"],
                str(src.get("page", "-")),
                f"{src['relevance_score']:.3f}",
            )
        console.print(table)


@app.command()
def chat():
    """대화형 멀티턴 Q&A 세션을 시작합니다. 종료: 'exit' 또는 Ctrl+C"""
    agent = _get_agent()

    if agent.status()["indexed_chunks"] == 0:
        console.print("[yellow]경고: 인덱싱된 문서가 없습니다. 먼저 `medrag ingest <경로>`를 실행하세요.[/yellow]")
        raise typer.Exit(1)

    console.print(Panel(
        "[bold cyan]MedRAG Agent[/bold cyan] — 대화형 모드\n"
        "종료하려면 [bold]exit[/bold] 또는 [bold]Ctrl+C[/bold]를 입력하세요.",
        border_style="cyan",
    ))

    while True:
        try:
            question = console.input("\n[bold green]질문[/bold green]: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]대화를 종료합니다.[/yellow]")
            break

        if question.lower() in ("exit", "quit", "종료"):
            console.print("[yellow]대화를 종료합니다.[/yellow]")
            break
        if not question:
            continue

        with console.status("[bold green]처리 중..."):
            result = agent.chat(question)

        confidence_color = {"high": "green", "medium": "yellow", "low": "red"}.get(
            result.confidence, "white"
        )
        console.print(
            Panel(
                Markdown(result.answer),
                title=f"[bold]답변[/bold] | 신뢰도: [{confidence_color}]{result.confidence}[/{confidence_color}]",
                border_style="blue",
            )
        )


@app.command()
def status():
    """현재 벡터 DB 및 Agent 상태를 출력합니다."""
    agent = _get_agent()
    info = agent.status()

    table = Table(title="MedRAG Agent 상태", show_header=False)
    table.add_column("항목", style="cyan")
    table.add_column("값", style="white")
    for key, value in info.items():
        table.add_row(key, str(value))
    console.print(table)


if __name__ == "__main__":
    app()
