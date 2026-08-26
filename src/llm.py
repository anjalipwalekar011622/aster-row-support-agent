"""
Wraps the Groq API call so agent.py doesn't need to know the request/response
format details.
"""

from __future__ import annotations
import json
import os
import requests
import re


class GroqLLM:
    MODEL = "openai/gpt-oss-20b"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set")

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        import time

        max_retries = 5
        for attempt in range(max_retries):
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.MODEL,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "temperature": 0.1,
                    "max_tokens": 400,
                },
                timeout=30,
            )
            if resp.status_code == 429:
                # Rate limited -- Groq tells us how long to wait, so respect that
                # instead of guessing. Add a small buffer on top to be safe.
                wait_seconds = 2.0
                try:
                    error_body = resp.json()
                    msg = error_body.get("error", {}).get("message", "")
                    # message looks like "...try again in 1.2375s..."
                    import re
                    m = re.search(r"try again in ([\d.]+)s", msg)
                    if m:
                        wait_seconds = float(m.group(1)) + 0.5
                except Exception:
                    pass
                print(f"[rate limited, waiting {wait_seconds:.1f}s before retry {attempt + 1}/{max_retries}]")
                time.sleep(wait_seconds)
                continue

            if resp.status_code != 200:
                print("GROQ ERROR BODY:", resp.text)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]["message"]
            tool_calls = []
            for tc in choice.get("tool_calls") or []:
                tool_calls.append(
                    {
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": json.loads(tc["function"]["arguments"]),
                    }
                )
            return {"content": choice.get("content"), "tool_calls": tool_calls}

        raise RuntimeError("Groq rate limit retries exhausted")


class FakeLLM:
    """Deterministic offline double used to exercise retrieval and tool plumbing.

    It deliberately does not pretend to be a quality substitute for a real
    model; its purpose is to let the evaluation harness run without a Groq
    credential and to make tool behaviour reproducible during development.
    """

    _ORDER_ID_RE = re.compile(r"ORD-?\s?\d{4,}|order\s+(?:id\s+)?\d{4,}", re.IGNORECASE)
    _PRIVACY_TERMS = ("email", "address", "internal note", "risk score", "phone number", "credit card")

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

        if messages and messages[-1]["role"] == "tool":
            result = json.loads(messages[-1]["content"])
            if result.get("acknowledged"):
                # A handoff acknowledgement follows the original order result.
                # Keep that result available so the user still receives the
                # useful, safe explanation (not just a generic handoff).
                prior_order = next(
                    (json.loads(m["content"]) for m in reversed(messages[:-1])
                     if m["role"] == "tool" and "found" in json.loads(m["content"])),
                    None,
                )
                if prior_order and not prior_order.get("found"):
                    return {"content": "I couldn't find an order with that ID. Please double-check the order ID or contact support. I've flagged this for a human specialist.", "tool_calls": []}
                if prior_order and prior_order.get("status") == "exception":
                    return {"content": f"Order {prior_order['order_id']} has a shipping exception and needs support review. I've flagged this for a human specialist.", "tool_calls": []}
                return {"content": "I've flagged this for a human specialist to follow up.", "tool_calls": []}
            if not result.get("found"):
                return {
                    "content": None,
                    "tool_calls": [{"id": "fake-handoff", "name": "flag_for_human_handoff", "arguments": {"reason": "order lookup did not resolve cleanly"}}],
                }
            if result.get("status") == "exception":
                return {
                    "content": None,
                    "tool_calls": [{"id": "fake-handoff", "name": "flag_for_human_handoff", "arguments": {"reason": "order has a shipping exception"}}],
                }
            if result["status"] == "cancelled":
                return {"content": f"Order {result['order_id']} is cancelled and will not be shipped.", "tool_calls": []}
            if result["status"] == "shipped" and not result.get("estimated_delivery"):
                return {"content": f"Order {result['order_id']} has shipped with {result.get('carrier')}. A delivery estimate is unavailable.", "tool_calls": []}
            eta = result.get("estimated_delivery")
            if eta and re.fullmatch(r"\d{4}-\d{2}-\d{2}", eta):
                from datetime import date
                eta = date.fromisoformat(eta).strftime("%B %d, %Y").replace(" 0", " ")
            return {"content": f"Order {result['order_id']} is {result['status']} with {result.get('carrier')}, estimated to arrive {eta}.", "tool_calls": []}

        if any(term in last_user.lower() for term in self._PRIVACY_TERMS):
            return {
                "content": None,
                "tool_calls": [{"id": "fake-privacy", "name": "flag_for_human_handoff", "arguments": {"reason": "request for private customer data"}}],
            }

        order_id = self._ORDER_ID_RE.search(last_user)
        if order_id:
            return {"content": None, "tool_calls": [{"id": "fake-lookup", "name": "order_lookup", "arguments": {"order_id": order_id.group(0)}}]}
        if any(word in last_user.lower() for word in ("where is my order", "track my order", "when will my order")):
            return {"content": "Please share your order ID (for example, ORD-1234) so I can look it up.", "tool_calls": []}

        context = next((m["content"] for m in reversed(messages) if m["role"] == "system" and "RETRIEVED_CONTEXT" in m["content"]), "")
        return {"content": f"[Offline retrieval preview]\n{context[:1200]}", "tool_calls": []}
