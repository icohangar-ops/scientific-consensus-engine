"""Synthesize CHP debate sessions into research briefs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chp import CHPSession
from nebius_client import ORCHESTRATOR_MODEL, NebiusAgent


def resolve_session(session_arg: str, sessions_root: Path) -> Path:
    if session_arg == "latest":
        candidates = sorted(sessions_root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError("No sessions found. Run pipeline.py first.")
        return candidates[0]
    path = Path(session_arg)
    if not path.exists():
        raise FileNotFoundError(f"Session not found: {path}")
    return path


def load_chp_session(session_dir: Path) -> CHPSession:
    chp_path = session_dir / "chp_session.json"
    if not chp_path.exists():
        raise FileNotFoundError(f"No debate session at {chp_path}. Run debate.py first.")
    with open(chp_path, encoding="utf-8") as f:
        return CHPSession.from_dict(json.load(f))


def synthesize_brief(session: CHPSession, mock: bool = False) -> dict:
    if mock or not __import__("os").environ.get("NEBIUS_API_KEY"):
        return {
            "hypothesis": session.hypothesis,
            "chp_state": session.state.value,
            "confidence_score": max(session.evidence_weight, 0.0),
            "novelty_index": session.novelty_index or 0.74,
            "evidence_map": {
                "supporting": [c.claim for c in session.claims if c.stance == "supporting"],
                "contradicting": [c.claim for c in session.claims if c.stance == "contradicting"],
            },
            "drug_target_candidates": ["TRAF2", "IL6", "NFKB1"],
            "gap_analysis": "Mechanistic validation in patient-derived CAR-T models still required.",
            "recommended_next_experiments": [
                "CRISPR perturbation of TRAF2 in CAR-T co-culture",
                "IL-6 / tocilizumab combination arm in resistant solid tumor model",
            ],
            "citations": [c.pmid for c in session.claims if c.pmid],
            "evidence_weight": session.evidence_weight,
            "integrity_hash": session.integrity_hash,
            "mode": "mock",
        }

    agent = NebiusAgent(
        model=ORCHESTRATOR_MODEL,
        system_prompt=(
            "You synthesize scientific debate outcomes into an investor-grade research brief. "
            "Return JSON with: hypothesis, chp_state, confidence_score (0-1), novelty_index (0-1), "
            "evidence_map (supporting vs contradicting summary), drug_target_candidates (list), "
            "gap_analysis, recommended_next_experiments, citations (pmid list)."
        ),
    )
    payload = json.dumps(session.to_dict(), indent=2)
    result = agent.chat(
        messages=[
            {
                "role": "user",
                "content": f"Synthesize this CHP debate session into a research brief:\n\n{payload}",
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=2500,
    )
    try:
        brief = json.loads(result.get("content", "{}"))
    except json.JSONDecodeError:
        brief = {"raw": result.get("content", "")}

    brief.setdefault("hypothesis", session.hypothesis)
    brief.setdefault("chp_state", session.state.value)
    brief.setdefault("evidence_weight", session.evidence_weight)
    brief.setdefault("integrity_hash", session.integrity_hash)
    return brief


def render_html(brief: dict) -> str:
    targets = brief.get("drug_target_candidates", [])
    targets_html = "".join(f"<li>{t}</li>" for t in targets) if targets else "<li>None identified</li>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Research Brief - Scientific Consensus Engine</title>
  <style>
    body {{ font-family: Georgia, serif; max-width: 860px; margin: 2rem auto; line-height: 1.6; }}
    h1 {{ font-size: 1.6rem; }}
    .meta {{ color: #555; font-size: 0.95rem; }}
    .badge {{ display: inline-block; padding: 0.2rem 0.6rem; background: #eef; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Scientific Consensus Engine - Research Brief</h1>
  <p class="meta"><span class="badge">{brief.get('chp_state', 'UNKNOWN')}</span>
     Confidence: {brief.get('confidence_score', brief.get('evidence_weight', 'N/A'))}
     | Novelty: {brief.get('novelty_index', 'N/A')}</p>
  <h2>Hypothesis</h2>
  <p>{brief.get('hypothesis', '')}</p>
  <h2>Evidence Map</h2>
  <p>{brief.get('evidence_map', brief.get('gap_analysis', ''))}</p>
  <h2>Drug Target Candidates</h2>
  <ul>{targets_html}</ul>
  <h2>Recommended Next Experiments</h2>
  <p>{brief.get('recommended_next_experiments', 'See gap analysis.')}</p>
  <p class="meta">Integrity hash: {brief.get('integrity_hash', 'pending')}</p>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize debate into research brief")
    parser.add_argument("--session", default="latest", help="Session dir or 'latest'")
    parser.add_argument("--format", choices=["json", "html"], default="json")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--mock", action="store_true", help="Offline brief without Nebius API")
    args = parser.parse_args()

    session_dir = resolve_session(args.session, ROOT / "sessions")
    chp_session = load_chp_session(session_dir)
    use_mock = args.mock or not __import__("os").environ.get("NEBIUS_API_KEY")
    brief = synthesize_brief(chp_session, mock=use_mock)

    if args.format == "html":
        content = render_html(brief)
        default_name = "report.html"
    else:
        content = json.dumps(brief, indent=2)
        default_name = "findings.json"

    out_path = Path(args.output) if args.output else session_dir / default_name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Brief written: {out_path}")


if __name__ == "__main__":
    main()
