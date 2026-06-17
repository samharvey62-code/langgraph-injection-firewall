"""Tests for the pluggable LLM factory's selection logic."""
import guard.llm as llm


def test_selects_openai_when_key_present(monkeypatch):
    calls = {}
    monkeypatch.setattr(llm, "_build_openai", lambda: (lambda p: calls.update({"openai": p}) or "openai-said"))
    monkeypatch.setattr(llm, "_build_local", lambda: (lambda p: "local-said"))
    fn = llm.build_generate_fn(env={"OPENAI_API_KEY": "sk-test"})
    assert fn("hi") == "openai-said"
    assert calls["openai"] == "hi"


def test_selects_local_when_key_absent(monkeypatch):
    monkeypatch.setattr(llm, "_build_openai", lambda: (lambda p: "openai-said"))
    monkeypatch.setattr(llm, "_build_local", lambda: (lambda p: "local-said"))
    fn = llm.build_generate_fn(env={})
    assert fn("hi") == "local-said"


def test_empty_key_falls_back_to_local(monkeypatch):
    monkeypatch.setattr(llm, "_build_openai", lambda: (lambda p: "openai-said"))
    monkeypatch.setattr(llm, "_build_local", lambda: (lambda p: "local-said"))
    fn = llm.build_generate_fn(env={"OPENAI_API_KEY": ""})
    assert fn("hi") == "local-said"
