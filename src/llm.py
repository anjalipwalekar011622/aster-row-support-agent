"""
Wraps the Groq API call so agent.py doesn't need to know the request/response
format details.
"""

from __future__ import annotations
import json
import os
import requests


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