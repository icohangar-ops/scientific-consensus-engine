"""Consensus Hardening Protocol (CHP) state machine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class CHPState(str, Enum):
    R0_GATE = "R0_GATE"
    EXPLORING = "EXPLORING"
    PROVISIONAL_LOCK = "PROVISIONAL_LOCK"
    LOCKED = "LOCKED"


@dataclass
class EvidenceClaim:
    claim: str
    stance: str  # supporting | contradicting | neutral
    pmid: str | None = None
    source_title: str | None = None
    confidence: float = 0.0


@dataclass
class CHPSession:
    hypothesis: str
    state: CHPState = CHPState.R0_GATE
    r0_passed: bool = False
    attack_vectors: list[str] = field(default_factory=list)
    claims: list[EvidenceClaim] = field(default_factory=list)
    evidence_weight: float = 0.0
    novelty_index: float = 0.0
    debate_rounds: list[dict[str, Any]] = field(default_factory=list)
    audit_trail: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    integrity_hash: str | None = None

    def log(self, event: str, details: dict[str, Any] | None = None) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": self.state.value,
            "event": event,
        }
        if details:
            entry["details"] = details
        self.audit_trail.append(entry)

    def run_r0_gate(self, prior_art_summary: str, attack_vectors: list[str]) -> bool:
        self.attack_vectors = attack_vectors
        passed = bool(prior_art_summary.strip()) and len(attack_vectors) >= 1
        self.r0_passed = passed
        self.log(
            "r0_gate",
            {
                "passed": passed,
                "prior_art_summary": prior_art_summary[:500],
                "attack_vectors": attack_vectors,
            },
        )
        if passed:
            self.state = CHPState.EXPLORING
        return passed

    def add_claims(self, claims: list[EvidenceClaim]) -> None:
        self.claims.extend(claims)
        self._recompute_evidence_weight()

    def _recompute_evidence_weight(self) -> None:
        if not self.claims:
            self.evidence_weight = 0.0
            return
        supporting = sum(1 for c in self.claims if c.stance == "supporting")
        contradicting = sum(1 for c in self.claims if c.stance == "contradicting")
        total = len(self.claims)
        self.evidence_weight = round((supporting - contradicting) / total, 3)

    def record_debate_round(self, round_number: int, agent_outputs: dict[str, Any]) -> None:
        self.debate_rounds.append(
            {"round": round_number, "agents": agent_outputs, "evidence_weight": self.evidence_weight}
        )
        self.log("debate_round", {"round": round_number, "evidence_weight": self.evidence_weight})

    def try_provisional_lock(self, threshold: float = 0.7) -> bool:
        if self.state != CHPState.EXPLORING:
            return False
        if self.evidence_weight >= threshold:
            self.state = CHPState.PROVISIONAL_LOCK
            self._stamp_integrity()
            self.log("provisional_lock", {"evidence_weight": self.evidence_weight})
            return True
        return False

    def approve_lock(self, reviewer: str = "human") -> bool:
        if self.state != CHPState.PROVISIONAL_LOCK:
            return False
        self.state = CHPState.LOCKED
        self._stamp_integrity()
        self.log("locked", {"reviewer": reviewer})
        return True

    def _stamp_integrity(self) -> None:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        self.integrity_hash = hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CHPSession":
        claims = [EvidenceClaim(**c) for c in data.get("claims", [])]
        session = cls(
            hypothesis=data["hypothesis"],
            state=CHPState(data.get("state", CHPState.R0_GATE.value)),
            r0_passed=data.get("r0_passed", False),
            attack_vectors=data.get("attack_vectors", []),
            claims=claims,
            evidence_weight=data.get("evidence_weight", 0.0),
            novelty_index=data.get("novelty_index", 0.0),
            debate_rounds=data.get("debate_rounds", []),
            audit_trail=data.get("audit_trail", []),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            integrity_hash=data.get("integrity_hash"),
        )
        return session
