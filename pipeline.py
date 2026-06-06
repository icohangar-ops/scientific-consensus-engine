"""Ingest scientific literature into a session context store."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agents.context_engine import ContextEngine
from tools.literature_search import search_pubmed


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "session"


def create_session_dir(topic: str, base: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    session_dir = base / f"{stamp}-{slugify(topic)}"
    session_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "topic": topic,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(session_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return session_dir


def ingest_topic(topic: str, max_papers: int, session_dir: Path | None = None) -> Path:
    sessions_root = ROOT / "sessions"
    session_dir = session_dir or create_session_dir(topic, sessions_root)

    print(f"Ingesting literature: {topic}")
    papers = search_pubmed(topic, max_results=max_papers)
    engine = ContextEngine(session_dir)
    added = engine.add_papers(papers, embed=True)
    print(f"  Papers stored: {len(engine.papers)} (+{added} new)")
    return session_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest papers for Scientific Consensus Engine")
    parser.add_argument("--topic", required=True, help="Research topic to ingest")
    parser.add_argument("--max-papers", type=int, default=20, help="Maximum papers to fetch")
    parser.add_argument(
        "--session",
        default=None,
        help="Existing session directory (optional)",
    )
    parser.add_argument(
        "--sources",
        default="pubmed",
        help="Comma-separated sources (pubmed supported; arxiv/biorxiv reserved)",
    )
    args = parser.parse_args()
    _ = args.sources

    session_path = Path(args.session) if args.session else None
    session_dir = ingest_topic(args.topic, args.max_papers, session_path)
    print(f"Session: {session_dir}")


if __name__ == "__main__":
    main()
