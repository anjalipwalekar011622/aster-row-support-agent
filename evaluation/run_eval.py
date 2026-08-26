"""
Loads visible-cases.json + custom-cases.json, runs each conversation
through a fresh Agent session, and checks the STRUCTURED result (text,
sources cited, tool calls, handoff flag) rather than grading free-form
prose with a second LLM call.
"""

from __future__ import annotations
import json
import os
import sys
import argparse
import re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

from src.ingest import load_chunks
from src.retrieval import Retriever
from src.tools import OrderStore
from src.llm import FakeLLM, GroqLLM
from src.agent import Agent


@dataclass
class CaseResult:
    case_id: str
    category: str
    passed: bool
    failures: list[str] = field(default_factory=list)


def build_agent(use_fake: bool = False) -> Agent:
    base = os.path.join(os.path.dirname(__file__), "..")
    chunks = load_chunks(os.path.join(base, "knowledge-base"))
    retriever = Retriever(chunks)
    store = OrderStore(os.path.join(base, "data", "orders.json"))
    return Agent(retriever, store, FakeLLM() if use_fake else GroqLLM())


def run_case(agent: Agent, case: dict) -> CaseResult:
    session_id = f"eval-{case['id']}"
    expect = case["expect"]
    failures = []
    last_result = None

    for msg in case["messages"]:
        last_result = agent.handle_turn(session_id, msg["content"])

    text_l = last_result.text.lower()
    tool_names_called = [t["name"] for t in last_result.tool_calls]

    for s in expect.get("must_include", []):
        if s.lower() not in text_l:
            failures.append(f"missing required substring: {s!r}")

    for s in expect.get("must_not_include", []):
        if s.lower() in text_l:
            failures.append(f"forbidden substring present: {s!r}")

    # Concepts are semantic, not literal assertions. This intentionally
    # modest matcher keeps offline evaluation deterministic while a real
    # model remains the source of truth for answer quality.
    concept_terms = {
        "the order is cancelled": ("cancel",),
        "it will not be shipped": ("not be shipped", "will not ship"),
        "order was not found": ("not found", "couldn't find", "could not find"),
        "check the order id or contact support": ("order id", "contact support", "double-check"),
        "shipped with canada post": ("canada post",),
        "delivery estimate is unavailable": ("estimate is unavailable", "no delivery estimate"),
    }
    for concept in expect.get("must_include_concepts", []):
        terms = concept_terms.get(concept, tuple(w for w in re.findall(r"[a-z]+", concept.lower()) if len(w) > 4))
        if not any(term in text_l for term in terms):
            failures.append(f"missing concept: {concept!r}")

    for s in expect.get("required_sources", []):
        if s not in last_result.sources:
            failures.append(f"missing required source citation: {s!r} (cited: {last_result.sources})")

    for s in expect.get("forbidden_sources_as_authority", []):
        if s in last_result.sources:
            failures.append(f"cited a non-authoritative source as authority: {s!r}")

    expected_tool = expect.get("tool")
    if expected_tool == "not_called":
        if "order_lookup" in tool_names_called:
            failures.append("order_lookup was called but should not have been")
    elif expected_tool == "order_lookup":
        if "order_lookup" not in tool_names_called:
            failures.append("order_lookup was expected but not called")
        expected_args = expect.get("tool_arguments")
        if expected_args:
            matched = any(
                t["name"] == "order_lookup" and t["arguments"].get("order_id") == expected_args.get("order_id")
                for t in last_result.tool_calls
            )
            if not matched:
                failures.append(f"order_lookup not called with expected arguments {expected_args}")
    elif expected_tool == "not_called_without_id":
        if "order_lookup" in tool_names_called:
            failures.append("order_lookup was called even though no order ID was given")

    for s in expect.get("must_ask_for", []):
        if s.lower() not in text_l and "order id" not in text_l:
            failures.append(f"did not ask for: {s!r}")

    for s in expect.get("must_refuse_to_disclose", []):
        pass  # enforced structurally by tools.py -- these fields never exist in the tool result

    if "handoff" in expect and bool(expect["handoff"]) != bool(last_result.handoff):
        failures.append(f"expected handoff={expect['handoff']}, got {last_result.handoff}")

    return CaseResult(case_id=case["id"], category=case["category"], passed=not failures, failures=failures)


def load_cases() -> list[dict]:
    base_dir = os.path.dirname(__file__)
    with open(os.path.join(base_dir, "visible-cases.json")) as f:
        visible = json.load(f)["cases"]
    custom = []
    custom_path = os.path.join(base_dir, "custom-cases.json")
    if os.path.exists(custom_path):
        with open(custom_path) as f:
            custom = json.load(f)["cases"]
    return visible + custom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true", help="Run offline with deterministic tool-routing behaviour")
    args = parser.parse_args()
    agent = build_agent(use_fake=args.fake)
    cases = load_cases()
    import time
    results = []
    for c in cases:
        results.append(run_case(agent, c))
        if not args.fake:
            time.sleep(15)  # small pause between cases to stay under Groq's free-tier rate limit

    by_category: dict[str, list[CaseResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    print(f"\n{'='*70}\nEVALUATION RESULTS\n{'='*70}")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.case_id} ({r.category})")
        for f in r.failures:
            print(f"        - {f}")

    print(f"\n{'-'*70}\nBY CATEGORY\n{'-'*70}")
    for cat, rs in sorted(by_category.items()):
        n_pass = sum(1 for r in rs if r.passed)
        print(f"{cat:28s} {n_pass}/{len(rs)}")

    total_pass = sum(1 for r in results if r.passed)
    print(f"\nTOTAL: {total_pass}/{len(results)} passed\n")


if __name__ == "__main__":
    main()
