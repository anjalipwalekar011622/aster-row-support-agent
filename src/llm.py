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