"""Tests for graph routing, using a stub classifier and generate function."""
from guard.graph import build_graph
from guard.state import new_state


class StubClf:
    def __init__(self, score):
        self._score = score

    def score(self, text):
        return self._score


def _run(score, threshold=0.5):
    app = build_graph(StubClf(score), generate_fn=lambda p: "GENERATED", threshold=threshold)
    return app.invoke(new_state("some prompt"))


def test_safe_prompt_is_allowed_and_generated():
    out = _run(score=0.1)
    assert out["verdict"] == "allowed"
    assert out["response"] == "GENERATED"
    assert out["score"] == 0.1


def test_injection_prompt_is_blocked():
    out = _run(score=0.95)
    assert out["verdict"] == "blocked"
    assert "GENERATED" not in out["response"]
    assert out["log"]  # a verdict was logged


def test_threshold_boundary_blocks_at_or_above():
    # score == threshold must block (>= threshold is hostile)
    out = _run(score=0.5, threshold=0.5)
    assert out["verdict"] == "blocked"
