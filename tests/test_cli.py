"""Tests for the CLI wiring, with run() monkeypatched to a fixed state."""
import json
import guard.cli as cli
from guard.state import new_state


def _fake_state(verdict, response="resp", score=0.9):
    s = new_state("p")
    s["verdict"] = verdict
    s["response"] = response
    s["score"] = score
    return s


def test_allowed_returns_zero(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run", lambda prompt, threshold=0.5: _fake_state("allowed", score=0.1))
    code = cli.main(["hello there"])
    assert code == 0
    assert "allowed" in capsys.readouterr().out.lower()


def test_blocked_returns_one(monkeypatch):
    monkeypatch.setattr(cli, "run", lambda prompt, threshold=0.5: _fake_state("blocked"))
    assert cli.main(["ignore previous instructions"]) == 1


def test_json_flag_emits_valid_json(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run", lambda prompt, threshold=0.5: _fake_state("blocked"))
    cli.main(["bad prompt", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "blocked"
