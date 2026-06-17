"""Prompt-injection detection: an ML classifier plus a naive keyword baseline.

Why two detectors: the keyword filter exists only as the benchmark baseline. The
dissertation's headline finding was that keyword filters collapse against obfuscation;
the ML classifier is what this firewall actually relies on. Keeping both here lets the
benchmark compare them on identical inputs.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional

MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"

# Naive baseline only: a handful of literal injection phrases. Deliberately weak —
# it is meant to be defeated by obfuscation, which is the point the benchmark makes.
_KEYWORD_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) instructions",
    r"disregard (all |the )?(previous|prior|above)",
    r"reveal (the )?(system )?prompt",
    r"you are now",
    r"do anything now",
    r"developer mode",
]
_KEYWORD_RE = re.compile("|".join(_KEYWORD_PATTERNS), re.IGNORECASE)


def keyword_flag(text: str) -> bool:
    """Return True if the naive keyword baseline considers the text an injection."""
    return bool(_KEYWORD_RE.search(text))


class InjectionClassifier:
    """Wraps the Hugging Face classifier and returns an injection probability.

    The HF pipeline is injectable so tests run without downloading a model; in
    production it is built lazily on first use.
    """

    def __init__(self, pipeline: Optional[Callable[[str], List[dict]]] = None):
        self._pipeline = pipeline

    def _ensure_pipeline(self) -> Callable[[str], List[dict]]:
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline  # lazy: heavy import
            self._pipeline = hf_pipeline("text-classification", model=MODEL_NAME)
        return self._pipeline

    def score(self, text: str) -> float:
        """Injection probability in [0, 1]. Fails closed (1.0) on inference error.

        Model *load* failure is deliberately NOT caught here: a missing/broken model is
        a setup problem the operator must see, not a prompt to silently block.
        """
        pipe = self._ensure_pipeline()  # load errors propagate with the model name
        try:
            result = pipe(text)[0]
            label = str(result["label"]).upper()
            confidence = float(result["score"])
            # Models report the predicted label with its confidence; normalise both
            # label conventions to a single "probability this is an injection".
            if "INJECT" in label or label in ("LABEL_1", "UNSAFE"):
                return confidence
            return 1.0 - confidence
        except Exception:
            # Fail closed: if we cannot assess a prompt, treat it as hostile.
            return 1.0
