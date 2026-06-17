"""Validates the structure and balance of the labelled test set."""
import json
from pathlib import Path

FAMILIES = {"direct", "jailbreak", "indirect", "obfuscated", "benign"}


def _load():
    path = Path(__file__).resolve().parents[1] / "data" / "testset.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_rows_have_required_shape():
    rows = _load()
    assert len(rows) >= 30
    for r in rows:
        assert set(r) >= {"text", "family", "label", "note"}
        assert r["family"] in FAMILIES
        assert r["label"] in (0, 1)
        assert r["note"].strip()


def test_benign_rows_are_label_zero_others_one():
    for r in _load():
        assert r["label"] == (0 if r["family"] == "benign" else 1)


def test_every_family_present():
    assert {r["family"] for r in _load()} == FAMILIES
