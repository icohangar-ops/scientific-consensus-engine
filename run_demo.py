#!/usr/bin/env python3
"""End-to-end demo: ingest, debate, synthesize."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(cmd: list[str]) -> None:
    print(f"\n> {' '.join(cmd)}\n")
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Scientific Consensus Engine demo")
    parser.add_argument(
        "--hypothesis",
        default="TRAF2 mediates CAR-T resistance via NF-kB signaling leading to IL-6 secretion",
    )
    parser.add_argument("--topic", default="CAR-T therapy resistance TRAF2 NF-kB IL-6")
    parser.add_argument("--max-papers", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--mock", action="store_true", default=True)
    args = parser.parse_args()

    run(
        [
            PYTHON,
            "pipeline.py",
            "--topic",
            args.topic,
            "--max-papers",
            str(args.max_papers),
        ]
    )
    debate_cmd = [
        PYTHON,
        "debate.py",
        "--hypothesis",
        args.hypothesis,
        "--rounds",
        str(args.rounds),
    ]
    if args.mock:
        debate_cmd.append("--mock")
    run(debate_cmd)
    synth_cmd = [PYTHON, "synthesize.py", "--session", "latest", "--format", "html"]
    if args.mock:
        synth_cmd.append("--mock")
    run(synth_cmd)
    print("\nDemo complete. Check sessions/ for papers, chp_session.json, and report.html")


if __name__ == "__main__":
    main()
