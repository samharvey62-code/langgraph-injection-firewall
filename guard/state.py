"""The single shared data contract that flows through the LangGraph pipeline.

Why one TypedDict: every node reads and writes the same state object, so a single
explicit contract keeps node interfaces honest and makes the graph easy to test.
"""
from __future__ import annotations

from typing import Dict, List, TypedDict


class GuardState(TypedDict):
    prompt: str          # the incoming user prompt under inspection
    score: float         # injection probability in [0, 1] from the classifier
    verdict: str         # "" until decided, then "allowed" or "blocked"
    response: str        # LLM output (allowed) or refusal text (blocked)
    log: List[Dict]      # ordered audit trail of what each node decided


def new_state(prompt: str) -> GuardState:
    """Build the initial state for a prompt before any node has run."""
    return GuardState(prompt=prompt, score=0.0, verdict="", response="", log=[])
