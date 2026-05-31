# 🔬 Scientific Consensus Engine

**NextGen BioAgents — Nucleate NYC BioHack 2026**
[![Nebius](https://img.shields.io/badge/Powered%20by-Nebius%20Token%20Factory-6B46C1)](https://tokenfactory.nebius.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> A multi-agent system that ingests scientific literature, debates hypotheses through adversarial consensus, and surfaces novel drug targets. Powered by **Nebius Token Factory** + **Consensus Hardening Protocol (CHP)**.

---

## 📋 Table of Contents
- [Challenge Track](#-challenge-track-research-agent)
- [Features](#-features)
- [Architecture](#-architecture)
- [How It Uses Nebius](#-how-it-uses-nebius)
- [Consensus Hardening Protocol](#-consensus-hardening-protocol-chp)
- [Setup Guide](#-setup-guide)
- [Command Reference](#-command-reference)
- [Demo Script](#-demo-video-3-min)
- [Troubleshooting](#-troubleshooting)
- [Judge Evaluation Criteria](#-judge-evaluation-criteria)

---

## 🏆 Challenge Track: Research Agent

> *"Agents that continuously ingest scientific papers, patents, and preprints; identify gaps in existing knowledge; generate novel hypotheses; and update their conclusions as new data emerges."*

### Why This Matters
- **2M+ papers** published per year — doubling every 12 years
- No human can read more than **~300 papers/year** with full comprehension
- Hidden connections between research fields are **systematically missed**
- **68% of preclinical research** is irreproducible — we need automated evidence validation

### Our Solution
A multi-agent debate arena where specialized agents argue for and against a scientific hypothesis using real literature. The Consensus Hardening Protocol governs the debate, ensuring traceable, auditable conclusions that surface drug targets humans would miss.

---

## ✨ Features

| Capability | Technical Detail |
|---|---|
| **Literature Ingestion** | PubMed, arXiv, bioRxiv fetch via Nebius function calling |
| **Adversarial Hypothesis Debate** | 3 agents (Optimist, Skeptic, Validator) on DeepSeek V3.2 |
| **CHP State Machine** | `R0 Gate → EXPLORING → PROVISIONAL_LOCK → LOCKED` |
| **Novelty Detection** | Cross-field connections using Qwen3 embedding similarity |
| **Evidence Citations** | Every claim traced to a paper with PMID/arXiv ID |
| **Live Updating** | Re-runs on new publications in a topic area |

---

## 🏗 Architecture

```
                    ┌──────────────────────────────────────────────────┐
  Literature ──────>│           Context Engine (PGVector)              │
  PubMed, arXiv     │  Papers embedded via Qwen3-Embedding-8B (4096d)  │
  bioRxiv, patents  │  Structured claims extracted by DeepSeek V3.2   │
                    └────────────────────┬─────────────────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────────────────┐
                    │            Multi-Agent Debate Arena              │
                    │  "Is TRAF2 a mediator of CAR-T resistance via    │
                    │   NF-κB → IL-6 signaling?"                      │
                    │                                                  │
         ┌──────────┴─────────┐  ┌──────────┴─────────┐  ┌──────────┴─────────┐
         │   Optimist         │  │   Skeptic          │  │   Validator        │
         │   Agent            │  │   Agent            │  │   Agent            │
         │                    │  │                    │  │                    │
         │ Finds supporting   │  │ Challenges the     │  │ Cross-checks all   │
         │ evidence           │  │ mechanism, finds   │  │ citations, runs    │
         │ Builds case        │  │ contradictions     │  │ plausibility       │
         └──────────┬─────────┘  └──────────┬─────────┘  └──────────┬─────────┘
                    │                       │                       │
                    └───────────────────────┼───────────────────────┘
                                            ▼
                    ┌──────────────────────────────────────────────────┐
                    │     Consensus Hardening Protocol (CHP)           │
                    │                                                  │
                    │  R0 Gate ──[pass]──> EXPLORING                   │
                    │                        │                        │
                    │                   [evidence >= 0.7]             │
                    │                        │                        │
                    │                   PROVISIONAL_LOCK               │
                    │                        │                        │
                    │                 [human review]                   │
                    │                        │                        │
                    │                   LOCKED                         │
                    └──────────────────────────────────────────────────┘
                                            │
                                            ▼
                    ┌──────────────────────────────────────────────────┐
                    │  Research Brief: Hypothesis + Evidence Map       │
                    │  + Confidence Score + Novelty Index              │
                    │  + Drug Target Candidates + Gap Analysis          │
                    └──────────────────────────────────────────────────┘
```

### Agent Profiles

| Agent | Persona | Goal | Evaluation Criteria |
|---|---|---|---|
| **Optimist** 🤖 | Supportive scientist | Find maximum supporting evidence | Tolerant — lower threshold for "evidence" |
| **Skeptic** 🕵️ | Rigorous reviewer | Challenge every claim | High bar — demands P<0.05, sample sizes, controls |
| **Validator** 🔬 | Meta-scientist | Cross-check all citations | Neutral — evaluates both sides, checks data integrity |

### Debate Protocol
```
Round 1: All agents read context papers from the Ingestion Engine
Round 2: Optimist presents supporting case with citations
Round 3: Skeptic challenges each claim with counter-evidence
Round 4: Validator fact-checks all citations and data integrity
Round 5: Each agent updates position (Bayesian)
   → CHP consolidation: assign confidence score
```

---

## 🔧 How It Uses Nebius Token Factory

| Capability | Nebius Model | Usage |
|---|---|---|
| **Adversarial Reasoning** | `deepseek/deepseek-chat-v3-2-0324` | Each debate agent uses deep reasoning |
| **Orchestration** | `meta-llama/Llama-3.3-70B-Instruct` | Route papers, manage debate, synthesize |
| **Literature Search** | Function calling → `search_pubmed()` | Real-time PubMed/arXiv queries |
| **Paper Embeddings** | `Qwen/Qwen3-Embedding-8B` | 4096-dim embeddings for RAG + similarity |
| **Structured JSON** | Response format enforcement | Deterministic debate records, evidence traces |

### Nebius Configuration
```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1",
    api_key=os.environ["NEBIUS_API_KEY"]
)

# Heavy reasoning for debate agents
response = client.chat.completions.create(
    model="deepseek/deepseek-chat-v3-2-0324",
    messages=[
        {"role": "system", "content": "You are the Optimist Agent. Find supporting evidence for the hypothesis."},
        {"role": "user", "content": "Hypothesis: TRAF2 mediates CAR-T resistance via NF-κB"}
    ],
    response_format={"type": "json_object"},
    temperature=0.3
)

# Embed papers for RAG similarity
emb_response = client.embeddings.create(
    model="Qwen/Qwen3-Embedding-8B",
    input=[paper_abstract]
)
print(f"Embedding dim: {len(emb_response.data[0].embedding)}")  # 4096
```

---

## 🔐 Consensus Hardening Protocol (CHP)

The CHP governs how scientific claims progress from "interesting idea" to "validated finding":

```
 R0 Gate ──────────────────────────────────────────────────
  • Foundation disclosure? (prior art check)
  • Potential attack vectors documented?
  • Adversarial case loaded with counter-evidence?

 ↓ [pass]

 EXPLORING ─────────────────────────────────────────────────
  • Optimist gathers supporting evidence from literature
  • Skeptic gathers counter-evidence from literature
  • Validator documents each claim with PMID citations
  • Evidence weight computed: (supporting - contradicting) / total

 ↓ [evidence_weight >= 0.7]

 PROVISIONAL_LOCK ──────────────────────────────────────────
  • Hypothesis tentatively accepted with documented gaps
  • All evidence, citations, and audit trail frozen
  • Awaiting human expert review

 ↓ [human reviewer approves]

 LOCKED ───────────────────────────────────────────────────
  • Fully validated claim
  • Hash-stamped for integrity verification
  • Added to knowledge base for future debate bootstrapping
```

---

## 🚀 Setup Guide

```bash
# 1. Clone
git clone https://github.com/zan-maker/scientific-consensus-engine
cd scientific-consensus-engine

# 2. Set up Nebius
export NEBIUS_API_KEY="nf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 3. Install
pip install -r requirements.txt

# 4. Ingest papers on a topic
python pipeline.py --topic "CAR-T therapy resistance mechanisms" --max-papers 100

# 5. Run debate
python debate.py --hypothesis "TRAF2 mediates CAR-T resistance via NF-kB signaling"

# 6. Generate research brief
python synthesize.py --session latest
```

---

## 📋 Command Reference

```bash
# Search and ingest papers from multiple sources
python pipeline.py --topic "checkpoint inhibitors PD-1 resistance" \
    --sources pubmed,arxiv,biorxiv --max-papers 500

# Run debate with custom temperature and rounds
python debate.py \
    --hypothesis "PD-L1 glycosylation affects immunotherapy response" \
    --temperature 0.2 --rounds 5

# Export consensus as JSON for external use
python synthesize.py --session latest --format json --output findings.json

# Generate HTML report
python synthesize.py --session latest --format html --output report.html

# Continuous monitoring (runs daily)
python monitor.py --topic "TRAF2 CAR-T resistance" \
    --check-interval 86400 --notify email
```

---

## 🎥 Demo Video (3 min)

| Time | Scene | Description |
|---|---|---|
| **0:00-0:30** | Problem | 2M+ papers/year — impossible to read. Hidden connections stay hidden. |
| **0:30-1:00** | Hypothesis Input | "TRAF2 mediates CAR-T resistance via NF-κB → IL-6" |
| **1:00-1:45** | Debate Rounds | Optimist finds evidence. Skeptic challenges. Validator cross-checks. All on DeepSeek V3.2 through Nebius |
| **1:45-2:15** | CHP Locks | EXPLORING → PROVISIONAL_LOCK → novel IL-6 connection discovered |
| **2:15-2:45** | Architecture | Cubiczan Swarm Pack + CHP on Nebius |
| **2:45-3:00** | CTA | Open source, extensible to any research domain |

---

## 🔧 Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| `429 Rate Limit` | Too many requests | Add `time.sleep(0.5)` between calls |
| Empty paper results | PubMed API availability | Check `pipeline.py` fallback mode |
| Low confidence in debate | Weak supporting literature | Ingest more papers with `--max-papers 500` |
| CHP stuck at EXPLORING | Not enough evidence | Increase debate rounds with `--rounds 7` |
| JSON parse error | Model temperature too high | Set `--temperature 0.2` or lower |

---

## 📊 Judge Evaluation Criteria

| Criterion | How We Address It |
|---|---|
| **Technical Complexity** | Multi-agent adversarial debate, CHP state machine, Nebius function calling + embeddings |
| **Impact & Feasibility** | Surfaces targets humans miss, auto-validates claims with citations |
| **Innovation** | Adversarial consensus for science — most systems validate, ours debates |
| **Presentation** | Demo video, architecture diagram, CHP flowchart in README |
| **Code Quality** | Modular agents, typed protocols, CEP-inspired state machine |

---

*Built at Nucleate NYC BioHack: NextGen BioAgents — June 6, 2026 · Automattic, 166 Crosby St, NYC*
*Powered by Nebius Token Factory · Sponsored by Nebius & Cursor*

⭐ *Star this repo if you find it useful! Contributions welcome.*
