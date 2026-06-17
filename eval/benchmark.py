"""Benchmark: does the ML classifier beat a naive keyword filter on obfuscation?

Why this exists: it is the credibility artifact and the direct corollary of the MSc
dissertation's headline finding (a 10-variant obfuscation probe defeated a keyword
filter 100% of the time). Here we show the defensive side — the ML classifier holds up
on the obfuscated family where the keyword baseline collapses.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List

from guard.classifier import keyword_flag

DATASET = Path(__file__).resolve().parents[1] / "data" / "testset.jsonl"


def load_dataset(path=DATASET) -> List[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evaluate(rows: List[dict], score_fn: Callable[[str], float], threshold: float = 0.5) -> Dict:
    """Tally detections per family for both detectors, plus benign false-positive rate."""
    summary: Dict[str, Dict] = {"ml": {}, "keyword": {}}
    families = sorted({r["family"] for r in rows})
    for det in summary:
        for fam in families:
            summary[det][fam] = {"detected": 0, "total": 0}

    benign_total = benign_ml_fp = benign_kw_fp = 0

    for r in rows:
        fam, is_injection = r["family"], r["label"] == 1
        ml_flag = score_fn(r["text"]) >= threshold
        kw_flag = keyword_flag(r["text"])

        if is_injection:
            summary["ml"][fam]["total"] += 1
            summary["keyword"][fam]["total"] += 1
            summary["ml"][fam]["detected"] += int(ml_flag)
            summary["keyword"][fam]["detected"] += int(kw_flag)
        else:  # benign: any flag is a false positive
            benign_total += 1
            benign_ml_fp += int(ml_flag)
            benign_kw_fp += int(kw_flag)

    summary["ml"]["false_positive_rate"] = (benign_ml_fp / benign_total) if benign_total else 0.0
    summary["keyword"]["false_positive_rate"] = (benign_kw_fp / benign_total) if benign_total else 0.0
    return summary


def format_table(summary: Dict) -> str:
    """Render a captioned detection-rate table comparing the two detectors."""
    # Exclude "false_positive_rate" (not a family) and "benign" (no injections to detect;
    # benign is already reported on the false-positive-rate line below).
    families = [f for f in summary["ml"] if f not in ("false_positive_rate", "benign")]
    lines = [
        "Detection rate by family (higher = better; obfuscated is the telling column)",
        "-" * 64,
        f"{'family':<14}{'ML classifier':<18}{'keyword filter':<18}",
    ]
    for fam in families:
        ml, kw = summary["ml"][fam], summary["keyword"][fam]
        ml_s = f"{ml['detected']}/{ml['total']}" if ml["total"] else "-"
        kw_s = f"{kw['detected']}/{kw['total']}" if kw["total"] else "-"
        lines.append(f"{fam:<14}{ml_s:<18}{kw_s:<18}")
    lines.append("-" * 64)
    lines.append(
        f"benign false-positive rate  ML={summary['ml']['false_positive_rate']:.2f}  "
        f"keyword={summary['keyword']['false_positive_rate']:.2f}"
    )
    return "\n".join(lines)


def main(argv=None) -> int:  # pragma: no cover - exercised manually with the real model
    from guard.classifier import InjectionClassifier
    clf = InjectionClassifier()
    rows = load_dataset()
    summary = evaluate(rows, score_fn=clf.score)
    print(format_table(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
