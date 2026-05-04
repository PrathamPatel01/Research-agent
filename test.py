"""
Enhanced Research Digest Agent
================================
Improvements over baseline:
  - Multi-source search (DuckDuckGo + Wikipedia fallback)
  - Follow-up question suggestions
  - Markdown export with timestamps
  - Search history with deduplication
  - Retry logic with exponential backoff
  - Rich console UI
  - Confidence scoring
  - Graceful error handling throughout
"""

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.live import Live
from rich import box
import time
import os
import json

load_dotenv()

# ── Console ──────────────────────────────────────────────────────────────────
console = Console()

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# ── Search tools ─────────────────────────────────────────────────────────────
ddg = DuckDuckGoSearchRun()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))

# ── Prompts ───────────────────────────────────────────────────────────────────

search_query_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research assistant. Given a topic, return ONLY a concise "
        "search query optimised for a web search engine. Nothing else.",
    ),
    ("human", "Topic: {topic}"),
])

digest_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior research analyst. Given combined search results,
produce a structured digest in Markdown using EXACTLY this format:

## 📌 Summary
2–3 sentence overview.

## 🔑 Key Developments
- bullet 1
- bullet 2
- bullet 3 (up to 5 total)

## 🔭 What to Watch Next
1–2 sentences on emerging trends or open questions.

## 🎯 Confidence
Rate the quality of available sources as Low / Medium / High and explain in
one sentence.
""",
    ),
    ("human", "Topic: {topic}\n\nCombined Search Results:\n{results}"),
])

followup_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a curious research assistant. Given a digest, suggest exactly "
        "3 concise follow-up research questions the user might want to explore next. "
        "Return ONLY a numbered list, nothing else.",
    ),
    ("human", "Topic: {topic}\n\nDigest:\n{digest}"),
])

# ── Chains ────────────────────────────────────────────────────────────────────
query_chain   = search_query_prompt | llm
digest_chain  = digest_prompt | llm
followup_chain = followup_prompt | llm

# ── Helpers ───────────────────────────────────────────────────────────────────

def retry(fn, *args, retries: int = 3, base_delay: float = 1.5, **kwargs):
    """Call *fn* with exponential backoff on failure."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            console.print(
                f"[yellow]⚠  Attempt {attempt + 1} failed ({exc}). "
                f"Retrying in {wait:.1f}s…[/yellow]"
            )
            time.sleep(wait)


def gather_results(search_query: str, topic: str) -> tuple[str, list[str]]:
    """Run DuckDuckGo; fall back to Wikipedia if results are thin."""
    sources: list[str] = []

    with Live(Spinner("dots", text="Searching DuckDuckGo…"), refresh_per_second=10):
        try:
            ddg_results = retry(ddg.run, search_query)
            sources.append("DuckDuckGo")
        except Exception as exc:
            console.print(f"[red]DuckDuckGo failed: {exc}[/red]")
            ddg_results = ""

    combined = ddg_results

    # Supplement with Wikipedia when DDG results are short
    if len(ddg_results) < 500:
        with Live(Spinner("dots", text="Supplementing with Wikipedia…"), refresh_per_second=10):
            try:
                wiki_results = retry(wiki.run, topic)
                combined = f"{ddg_results}\n\n[Wikipedia]\n{wiki_results}"
                sources.append("Wikipedia")
            except Exception as exc:
                console.print(f"[yellow]Wikipedia fallback failed: {exc}[/yellow]")

    return combined, sources


def export_markdown(topic: str, digest: str, followups: str, sources: list[str]) -> Path:
    """Save digest + follow-ups to a timestamped Markdown file."""
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)

    safe_topic = "".join(c if c.isalnum() or c in " _-" else "_" for c in topic)[:50]
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath   = exports_dir / f"{timestamp}_{safe_topic}.md"

    content = (
        f"# Research Digest: {topic}\n"
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        f"*Sources: {', '.join(sources)}*\n\n"
        f"{digest}\n\n"
        f"---\n\n"
        f"## 💡 Follow-up Questions\n{followups}\n"
    )
    filepath.write_text(content, encoding="utf-8")
    return filepath


def show_history(history: list[dict]):
    """Render search history as a Rich table."""
    if not history:
        console.print("[dim]No history yet.[/dim]")
        return

    table = Table(title="🕑 Research History", box=box.SIMPLE_HEAVY, show_lines=True)
    table.add_column("#",      style="dim",    width=4)
    table.add_column("Topic",  style="cyan",   min_width=30)
    table.add_column("Time",   style="yellow", width=20)
    table.add_column("Sources",style="green",  width=20)

    for i, entry in enumerate(history, 1):
        table.add_row(
            str(i),
            entry["topic"],
            entry["timestamp"],
            ", ".join(entry["sources"]),
        )
    console.print(table)


# ── Core pipeline ─────────────────────────────────────────────────────────────

def research_digest(topic: str, history: list[dict]) -> None:
    """Run the full research pipeline for *topic*."""

    # Deduplicate
    seen_topics = {e["topic"].lower() for e in history}
    if topic.lower() in seen_topics:
        console.print(
            Panel(
                f"[yellow]You've already researched '[bold]{topic}[/bold]'.\n"
                "Showing a fresh digest anyway.[/yellow]",
                title="ℹ  Duplicate detected",
                border_style="yellow",
            )
        )

    console.rule(f"[bold cyan]Researching: {topic}[/bold cyan]")

    # 1. Generate search query
    with Live(Spinner("dots", text="Generating search query…"), refresh_per_second=10):
        search_query = retry(query_chain.invoke, {"topic": topic}).content.strip()
    console.print(f"[dim]🔍 Query:[/dim] [italic]{search_query}[/italic]")

    # 2. Fetch results
    combined_results, sources = gather_results(search_query, topic)

    if not combined_results.strip():
        console.print("[red]❌ Could not retrieve any results. Try a different topic.[/red]")
        return

    # 3. Generate digest
    with Live(Spinner("dots", text="Generating digest…"), refresh_per_second=10):
        digest = retry(digest_chain.invoke, {"topic": topic, "results": combined_results}).content

    # 4. Generate follow-up questions
    with Live(Spinner("dots", text="Generating follow-up questions…"), refresh_per_second=10):
        followups = retry(followup_chain.invoke, {"topic": topic, "digest": digest}).content

    # 5. Display
    console.print(
        Panel(
            Markdown(digest),
            title="📋 Research Digest",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )
    console.print(
        Panel(
            Markdown(followups),
            title="💡 Follow-up Questions",
            border_style="green",
            padding=(1, 2),
        )
    )

    # 6. Export
    export_path = export_markdown(topic, digest, followups, sources)
    console.print(f"[dim]💾 Saved to:[/dim] [underline]{export_path}[/underline]\n")

    # 7. Update history
    history.append({
        "topic":     topic,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sources":   sources,
        "query":     search_query,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    history: list[dict] = []

    console.print(
        Panel(
            "[bold]Research Digest Agent[/bold]\n"
            "[dim]Commands: [bold]quit[/bold] · [bold]history[/bold] · [bold]clear[/bold][/dim]",
            border_style="bright_magenta",
            padding=(1, 4),
        )
    )

    while True:
        try:
            topic = console.input("\n[bold green]📝 Topic:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold]👋 Goodbye![/bold]")
            break

        match topic.lower():
            case "quit" | "exit" | "q":
                console.print("[bold]👋 Goodbye![/bold]")
                break
            case "history" | "h":
                show_history(history)
            case "clear" | "c":
                history.clear()
                console.print("[green]✅ History cleared.[/green]")
            case "":
                console.print("[yellow]⚠  Please enter a topic.[/yellow]")
            case _:
                research_digest(topic, history)


if __name__ == "__main__":
    main()
