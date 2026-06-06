"""Offline mock debate for demos without Nebius API key."""

from __future__ import annotations

import json
from pathlib import Path

from chp import CHPSession, EvidenceClaim
from agents.context_engine import ContextEngine


def run_mock_debate(hypothesis: str, session_dir: Path, rounds: int = 2) -> CHPSession:
    engine = ContextEngine(session_dir)
    context_papers = engine.papers[:5]

    session = CHPSession(hypothesis=hypothesis)
    session.run_r0_gate(
        prior_art_summary=f"Found {len(context_papers)} papers on CAR-T resistance and TRAF2/NF-kB signaling.",
        attack_vectors=[
            "TRAF2 loss may cause lymphopenia limiting therapeutic window",
            "Resistance may be antigen-loss dominant rather than cytokine mediated",
            "IL-6 blockade timing may affect CAR-T expansion",
        ],
    )

    mock_claims_support = [
        EvidenceClaim(
            claim="TRAF2 knockdown reduced IL-6 and improved CAR-T persistence",
            stance="supporting",
            pmid="35212345",
            source_title="TRAF2 signaling in CAR-T cell exhaustion",
            confidence=0.82,
        ),
        EvidenceClaim(
            claim="IL-6 blockade restored cytotoxicity in NF-kB-high TME",
            stance="supporting",
            pmid="35198765",
            source_title="IL-6 blockade reverses CAR-T dysfunction",
            confidence=0.78,
        ),
        EvidenceClaim(
            claim="NF-kB inhibition reduced PD-1 and LAG-3 exhaustion markers",
            stance="supporting",
            pmid="34987654",
            source_title="NF-kB pathway inhibitors enhance adoptive cell therapy",
            confidence=0.76,
        ),
        EvidenceClaim(
            claim="CD28 costimulation requires TRAF2 for downstream NF-kB activation",
            stance="supporting",
            pmid="35212345",
            source_title="TRAF2 signaling in CAR-T cell exhaustion",
            confidence=0.8,
        ),
        EvidenceClaim(
            claim="Serum IL-6 elevation predicts CAR-T resistance in solid tumors",
            stance="supporting",
            pmid="35198765",
            source_title="IL-6 blockade reverses CAR-T dysfunction",
            confidence=0.77,
        ),
    ]
    mock_claims_contra = [
        EvidenceClaim(
            claim="Complete TRAF2 loss causes lymphopenia in preclinical models",
            stance="contradicting",
            pmid="34876543",
            source_title="Contradictory role of TRAF2 in T cell homeostasis",
            confidence=0.71,
        ),
    ]

    for round_num in range(1, rounds + 1):
        if round_num == 1:
            session.add_claims(mock_claims_support)
            session.add_claims(mock_claims_contra)
        session.record_debate_round(
            round_num,
            {
                "optimist": {
                    "summary": "TRAF2-NF-kB-IL-6 axis supports resistance hypothesis.",
                    "mode": "mock",
                },
                "skeptic": {
                    "summary": "TRAF2 targeting may be too toxic; alternative resistance modes dominate.",
                    "mode": "mock",
                },
                "validator": {
                    "summary": "Citations check out; mechanistic link plausible but not definitive.",
                    "novelty_notes": "IL-6 as bridge between TRAF2 and exhaustion is under-explored clinically.",
                    "mode": "mock",
                },
            },
        )
        if session.try_provisional_lock(threshold=0.65):
            break

    session.novelty_index = 0.74
    out = session_dir / "chp_session.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(session.to_dict(), f, indent=2)
    return session
