"""Run multi-agent adversarial debate under CHP governance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agents.context_engine import ContextEngine
from agents.debate_agents import run_optimist, run_r0_assessment, run_skeptic, run_validator
from agents.mock_debate import run_mock_debate
from chp import CHPSession
from pipeline import ingest_topic


def latest_session(sessions_root: Path) -> Path | None:
    sessions = sorted(sessions_root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return sessions[0] if sessions else None


def _public_agent_view(agent_output: dict) -> dict:
    return {k: v for k, v in agent_output.items() if not k.startswith("_")}


def run_debate(
    hypothesis: str,
    session_dir: Path,
    rounds: int = 3,
    lock_threshold: float = 0.7,
) -> CHPSession:
    engine = ContextEngine(session_dir)
    relevant = engine.top_k(hypothesis, k=min(8, len(engine.papers)))
    context = engine.context_block(relevant)

    session = CHPSession(hypothesis=hypothesis)
    print("CHP: R0 Gate...")
    r0 = run_r0_assessment(hypothesis, context)
    passed = session.run_r0_gate(
        prior_art_summary=r0.get("prior_art_summary", context[:500]),
        attack_vectors=r0.get("attack_vectors", ["Mechanism may be context-dependent"]),
    )
    if not passed:
        print("R0 gate failed: insufficient prior art documentation.")
        return session

    prior_summary = ""
    for round_num in range(1, rounds + 1):
        print(f"Debate round {round_num}/{rounds}...")
        optimist = run_optimist(hypothesis, context, prior_summary)
        skeptic = run_skeptic(
            hypothesis, context, json.dumps(_public_agent_view(optimist), indent=2)
        )
        validator = run_validator(
            hypothesis,
            context,
            json.dumps(_public_agent_view(optimist), indent=2),
            json.dumps(_public_agent_view(skeptic), indent=2),
        )

        session.add_claims(optimist.get("_evidence_claim_objects", []))
        session.add_claims(skeptic.get("_evidence_claim_objects", []))
        session.record_debate_round(
            round_num,
            {
                "optimist": _public_agent_view(optimist),
                "skeptic": _public_agent_view(skeptic),
                "validator": validator,
            },
        )
        prior_summary = (
            f"Round {round_num}\nOptimist: {optimist.get('summary', '')}\n"
            f"Skeptic: {skeptic.get('summary', '')}\n"
            f"Validator: {validator.get('summary', '')}"
        )
        print(f"  Evidence weight: {session.evidence_weight}")

        if session.try_provisional_lock(threshold=lock_threshold):
            print(f"CHP -> PROVISIONAL_LOCK (weight {session.evidence_weight})")
            break

    session_path = session_dir / "chp_session.json"
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(session.to_dict(), f, indent=2)
    print(f"Session saved: {session_path}")
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description="Scientific Consensus Engine - Debate")
    parser.add_argument("--hypothesis", required=True, help="Hypothesis to debate")
    parser.add_argument("--session", default=None, help="Session directory with ingested papers")
    parser.add_argument("--topic", default=None, help="Ingest topic if no session exists")
    parser.add_argument("--max-papers", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2, help="Reserved for future tuning")
    parser.add_argument("--lock-threshold", type=float, default=0.7)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline mock debate (no Nebius API required)",
    )
    args = parser.parse_args()
    _ = args.temperature

    sessions_root = ROOT / "sessions"
    if args.session:
        session_dir = Path(args.session)
    else:
        session_dir = latest_session(sessions_root)
        if session_dir is None:
            topic = args.topic or args.hypothesis
            print(f"No session found - ingesting: {topic}")
            session_dir = ingest_topic(topic, args.max_papers)

    if args.mock or not __import__("os").environ.get("NEBIUS_API_KEY"):
        if not args.mock and not __import__("os").environ.get("NEBIUS_API_KEY"):
            print("NEBIUS_API_KEY not set - running mock debate.")
        session = run_mock_debate(
            hypothesis=args.hypothesis,
            session_dir=session_dir,
            rounds=min(args.rounds, 3),
        )
        print(f"CHP state: {session.state.value} | evidence weight: {session.evidence_weight}")
        return

    run_debate(
        hypothesis=args.hypothesis,
        session_dir=session_dir,
        rounds=args.rounds,
        lock_threshold=args.lock_threshold,
    )


if __name__ == "__main__":
    main()
