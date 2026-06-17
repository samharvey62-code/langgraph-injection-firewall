"""Pluggable LLM for the safe path.

Why pluggable: a reviewer must be able to clone and run with zero API keys, so the
default is a small local Hugging Face model. Setting OPENAI_API_KEY transparently
upgrades the safe path to gpt-4o-mini (the dissertation's provider) without code
changes. Builders are lazy so importing this module never loads a model or needs a key.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, Optional

LOCAL_MODEL_NAME = "google/flan-t5-small"
OPENAI_MODEL_NAME = "gpt-4o-mini"


def _build_openai() -> Callable[[str], str]:
    from langchain_openai import ChatOpenAI
    chat = ChatOpenAI(model=OPENAI_MODEL_NAME)
    return lambda prompt: chat.invoke(prompt).content


def _build_local() -> Callable[[str], str]:
    from langchain_huggingface import HuggingFacePipeline
    pipe = HuggingFacePipeline.from_model_id(
        model_id=LOCAL_MODEL_NAME,
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": 256},
    )
    return lambda prompt: str(pipe.invoke(prompt))


def build_generate_fn(env: Optional[Dict[str, str]] = None) -> Callable[[str], str]:
    """Return a generate(prompt)->str function, OpenAI-backed if a key is set."""
    env = os.environ if env is None else env
    use_openai = bool(env.get("OPENAI_API_KEY"))
    builder = _build_openai if use_openai else _build_local
    built: Dict[str, Callable[[str], str]] = {}

    def generate(prompt: str) -> str:
        if "fn" not in built:
            built["fn"] = builder()  # lazy: only load a model when first asked
        return built["fn"](prompt)

    return generate
