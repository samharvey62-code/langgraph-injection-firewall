"""Command-line entry point: run one prompt through the firewall.

Why a thin CLI: it is the fastest way for a reviewer to see the firewall decide on a
prompt, and the exit code (0 allowed / 1 blocked) makes it scriptable.
"""
from __future__ import annotations

import argparse
import json
from typing import List, Optional

from guard.graph import run


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prompt-injection firewall (POC).")
    parser.add_argument("prompt", help="the prompt to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="injection score at/above which a prompt is blocked")
    parser.add_argument("--json", action="store_true", help="emit the full state as JSON")
    args = parser.parse_args(argv)

    state = run(args.prompt, threshold=args.threshold)

    if args.json:
        print(json.dumps(state, indent=2))
    else:
        print(f"verdict: {state['verdict']}  (injection score: {state['score']:.3f})")
        print(f"response: {state['response']}")

    return 0 if state["verdict"] == "allowed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
