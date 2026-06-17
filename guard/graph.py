"""The LangGraph pipeline that wires detection, routing, generation, and quarantine.

Why a graph: each concern is a separate node with one job, and the safe/unsafe decision
is an explicit conditional edge rather than a buried if-statement. Dependencies are
injected into build_graph() so the whole flow is unit-testable without loading models.
"""
from __future__ import annotations

from typing import Callable, Optional

from langgraph.graph import END, StateGraph

from guard.state import GuardState, new_state


def build_graph(classifier, generate_fn: Callable[[str], str], threshold: float = 0.5):
    """Compile the firewall graph. classifier needs .score(text)->float."""

    def classify_node(state: GuardState) -> dict:
        score = classifier.score(state["prompt"])
        return {
            "score": score,
            "log": state["log"] + [{"node": "classify", "score": score}],
        }

    def route(state: GuardState) -> str:
        # >= threshold is treated as hostile so the boundary fails safe.
        return "unsafe" if state["score"] >= threshold else "safe"

    def generate_node(state: GuardState) -> dict:
        response = generate_fn(state["prompt"])
        return {
            "verdict": "allowed",
            "response": response,
            "log": state["log"] + [{"node": "generate", "verdict": "allowed"}],
        }

    def quarantine_node(state: GuardState) -> dict:
        refusal = "Request blocked: prompt flagged as a possible injection attempt."
        return {
            "verdict": "blocked",
            "response": refusal,
            "log": state["log"] + [{"node": "quarantine", "verdict": "blocked",
                                    "score": state["score"]}],
        }

    graph = StateGraph(GuardState)
    graph.add_node("classify", classify_node)
    graph.add_node("generate", generate_node)
    graph.add_node("quarantine", quarantine_node)
    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", route,
                                {"safe": "generate", "unsafe": "quarantine"})
    graph.add_edge("generate", END)
    graph.add_edge("quarantine", END)
    return graph.compile()


def run(prompt: str, classifier=None, generate_fn: Optional[Callable[[str], str]] = None,
        threshold: float = 0.5) -> GuardState:
    """Run one prompt through the firewall, building real defaults if none given."""
    if classifier is None:
        from guard.classifier import InjectionClassifier
        classifier = InjectionClassifier()
    if generate_fn is None:
        from guard.llm import build_generate_fn
        generate_fn = build_generate_fn()
    app = build_graph(classifier, generate_fn, threshold=threshold)
    return app.invoke(new_state(prompt))
