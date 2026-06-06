"""Context engine: paper storage and embedding-backed retrieval."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from nebius_client import NebiusAgent


class ContextEngine:
    """Stores ingested papers and supports similarity retrieval."""

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.papers_path = self.session_dir / "papers.json"
        self.papers: list[dict[str, Any]] = []
        self.embeddings: list[list[float]] = []
        if self.papers_path.exists():
            self.load()

    def load(self) -> None:
        with open(self.papers_path, encoding="utf-8") as f:
            data = json.load(f)
        self.papers = data.get("papers", [])
        self.embeddings = data.get("embeddings", [])

    def save(self) -> None:
        with open(self.papers_path, "w", encoding="utf-8") as f:
            json.dump(
                {"papers": self.papers, "embeddings": self.embeddings},
                f,
                indent=2,
            )

    def add_papers(self, papers: list[dict[str, Any]], embed: bool = True) -> int:
        added = 0
        texts_to_embed: list[str] = []
        for paper in papers:
            if any(p.get("pmid") == paper.get("pmid") for p in self.papers):
                continue
            self.papers.append(paper)
            texts_to_embed.append(self._paper_text(paper))
            added += 1

        if embed and texts_to_embed and os.environ.get("NEBIUS_API_KEY"):
            agent = NebiusAgent()
            new_embeddings = agent.embed(texts_to_embed)
            self.embeddings.extend(new_embeddings)
        elif embed and texts_to_embed:
            # Deterministic pseudo-embeddings for offline demo
            for text in texts_to_embed:
                self.embeddings.append(self._pseudo_embed(text))

        self.save()
        return added

    def top_k(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if not self.papers:
            return []
        if not self.embeddings or not os.environ.get("NEBIUS_API_KEY"):
            return self.papers[:k]

        agent = NebiusAgent()
        query_vec = np.array(agent.embed([query])[0])
        scores = []
        for idx, emb in enumerate(self.embeddings):
            vec = np.array(emb)
            denom = np.linalg.norm(query_vec) * np.linalg.norm(vec)
            sim = float(np.dot(query_vec, vec) / denom) if denom else 0.0
            scores.append((sim, idx))
        scores.sort(reverse=True)
        return [self.papers[idx] for _, idx in scores[:k]]

    def context_block(self, papers: list[dict[str, Any]] | None = None) -> str:
        selected = papers or self.papers
        blocks = []
        for paper in selected:
            blocks.append(
                f"PMID:{paper.get('pmid')} | {paper.get('title')}\n"
                f"Abstract: {paper.get('abstract', '')[:800]}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _paper_text(paper: dict[str, Any]) -> str:
        return f"{paper.get('title', '')}. {paper.get('abstract', '')}"

    @staticmethod
    def _pseudo_embed(text: str, dim: int = 64) -> list[float]:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.standard_normal(dim)
        return vec.tolist()
