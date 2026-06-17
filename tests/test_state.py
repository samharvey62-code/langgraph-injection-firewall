"""Tests for the GuardState contract."""
from guard.state import GuardState, new_state


def test_new_state_initializes_all_fields():
    s = new_state("hello")
    assert s["prompt"] == "hello"
    assert s["score"] == 0.0
    assert s["verdict"] == ""
    assert s["response"] == ""
    assert s["log"] == []
