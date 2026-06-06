"""Debate agents: Optimist, Skeptic, Validator."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from chp import EvidenceClaim
from nebius_client import DEBATE_MODEL, NebiusAgent

OPTIMIST_PROMPT = """You are the Optimist Agent in a scientific debate arena.
Find the strongest supporting evidence for the hypothesis from the provided papers.
Cite PMIDs for every claim. Be constructive but evidence-bound.
Return JSON: {"position": "support", "summary": "...", "claims": [{"claim": "...", "pmid": "...", "source_title": "...", "confidence": 0.0-1.0}]}"""

SKEPTIC_PROMPT = """You are the Skeptic Agent in a scientific debate arena.
Challenge the hypothesis using contradicting evidence from the papers.
Demand methodological rigor (controls, sample size, reproducibility).
Return JSON: {"position": "challenge", "summary": "...", "claims": [{"claim": "...", "pmid": "...", "source_title": "...", "confidence": 0.0-1.0}]}"""

VALIDATOR_PROMPT = """You are the Validator Agent, a neutral meta-scientist.
Cross-check citations, flag unsupported leaps, and assess plausibility of both sides.
Return JSON: {"position": "validate", "summary": "...", "validated_claims": [...], "rejected_claims": [...], "novelty_notes": "..."}"""


def _parse_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content or "{}")
    except json.JSONDecodeError:
        return {"raw": content, "claims": []}


def _extract_claims(data: dict[str, Any], stance: str) -> list[EvidenceClaim]:
    claims = []
    for item in data.get("claims", []):
        claims.append(
            EvidenceClaim(
                claim=item.get("claim", ""),
                stance=stance,
                pmid=str(item.get("pmid", "")) or None,
                source_title=item.get("source_title"),
                confidence=float(item.get("confidence", 0.5)),
            )
        )
    return claims


def run_optimist(hypothesis: str, context: str, prior_rounds: str = "") -> dict[str, Any]:
    agent = NebiusAgent(model=DEBATE_MODEL, system_prompt=OPTIMIST_PROMPT)
    result = agent.chat(
        messages=[
            {
                "role": "user",
                "content": (
                    f"Hypothesis: {hypothesis}\n\nLiterature:\n{context}\n\n"
                    f"Prior debate:\n{prior_rounds}\n\nBuild the supporting case."
                ),
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    data = _parse_json(result.get("content", "{}"))
    claims = _extract_claims(data, "supporting")
    data["evidence_claims"] = [asdict(c) for c in claims]
    data["_evidence_claim_objects"] = claims
    return data


def run_skeptic(hypothesis: str, context: str, optimist_case: str) -> dict[str, Any]:
    agent = NebiusAgent(model=DEBATE_MODEL, system_prompt=SKEPTIC_PROMPT)
    result = agent.chat(
        messages=[
            {
                "role": "user",
                "content": (
                    f"Hypothesis: {hypothesis}\n\nLiterature:\n{context}\n\n"
                    f"Optimist case to challenge:\n{optimist_case}\n\n"
                    "Find contradictions and weaknesses."
                ),
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    data = _parse_json(result.get("content", "{}"))
    claims = _extract_claims(data, "contradicting")
    data["evidence_claims"] = [asdict(c) for c in claims]
    data["_evidence_claim_objects"] = claims
    return data


def run_validator(
    hypothesis: str, context: str, optimist_case: str, skeptic_case: str
) -> dict[str, Any]:
    agent = NebiusAgent(model=DEBATE_MODEL, system_prompt=VALIDATOR_PROMPT)
    result = agent.chat(
        messages=[
            {
                "role": "user",
                "content": (
                    f"Hypothesis: {hypothesis}\n\nLiterature:\n{context}\n\n"
                    f"Optimist:\n{optimist_case}\n\nSkeptic:\n{skeptic_case}\n\n"
                    "Validate citations and assess overall plausibility."
                ),
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return _parse_json(result.get("content", "{}"))


def run_r0_assessment(hypothesis: str, context: str) -> dict[str, Any]:
    agent = NebiusAgent(
        model=DEBATE_MODEL,
        system_prompt=(
            "You run the CHP R0 Gate. Summarize prior art and list attack vectors "
            "that could falsify the hypothesis. Return JSON with keys: "
            "prior_art_summary, attack_vectors (list of strings), pass (boolean)."
        ),
    )
    result = agent.chat(
        messages=[
            {
                "role": "user",
                "content": f"Hypothesis: {hypothesis}\n\nKnown literature:\n{context}",
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return _parse_json(result.get("content", "{}"))
